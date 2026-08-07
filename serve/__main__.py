# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
__main__.py — `python3 -m serve MODEL`.

Flags mirror the CLI's where they mean the same thing (--budget, --ctx,
--threads, --cpus, --vision), because a person who has run `waste run` should not
have to learn a second vocabulary to serve the same container.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, DecimalException
import os
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):                    # python3 serve/__main__.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "serve"

from . import api                                            # noqa: E402
from .engine import (CACHE_LFRU, CACHE_LRU,                  # noqa: E402
                     WASTE_E_ARG, WASTE_E_UNSUPPORTED,
                     Engine, EngineError, build_info, physical_ram,
                     usable_ram, memory_ceiling, plan_memory)
from .prefix_cache import CONTROLLER_OVERHEAD_BYTES          # noqa: E402
from .profiles import (PROFILE_NAMES, ProfileError,          # noqa: E402
                       resolve_profile)
from .qos import (DisabledRequestQos, QosError, RequestQos,  # noqa: E402
                  discard_control_from_env)
from .server import serve                                    # noqa: E402

POLICIES = {"lfru": CACHE_LFRU, "lru": CACHE_LRU}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def parse_size(text: str) -> int:
    """`8G`, `512M`, `1024` (bytes). The CLI's --budget spelling."""
    t = text.strip().upper()
    mult = 1
    if t.endswith(("K", "M", "G", "T")):
        mult = {"K": 1 << 10, "M": 1 << 20,
                "G": 1 << 30, "T": 1 << 40}[t[-1]]
        t = t[:-1]
    try:
        value = Decimal(t) * mult
    except DecimalException:
        raise argparse.ArgumentTypeError(f"not a size: {text}") from None
    if not value.is_finite() or value < 0 or value > (1 << 64) - 1:
        raise argparse.ArgumentTypeError(f"size is out of range: {text}")
    return int(value)


def bounded_int(lo: int, hi: int):
    def parse(text: str) -> int:
        try:
            value = int(text, 10)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not an integer: {text}") from None
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(
                f"integer must be between {lo} and {hi}: {text}")
        return value
    return parse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m serve",
        description="OpenAI-compatible server for a WASTE container.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python3 -m serve ~/models/k3.waste
  python3 -m serve ~/models/k3.waste --port 8080 --budget 48G --vision
  python3 -m serve ~/models/k3.waste --api-key "$WASTE_KEY" --host 0.0.0.0

  curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
    -d '{"model":"waste","messages":[{"role":"user","content":"hi"}]}'
""")
    ap.add_argument("model", help="path to the .waste container")
    ap.add_argument("--host", default="127.0.0.1",
                    help="default 127.0.0.1 — loopback only. Use 0.0.0.0 to "
                         "accept from the network, and set --api-key if you do")
    ap.add_argument("--port", type=bounded_int(0, 65535), default=8000)
    ap.add_argument("--model-id", default=None,
                    help="name reported by /v1/models (default: directory name)")
    ap.add_argument("--api-key", default=os.environ.get("WASTE_API_KEY"),
                    help="require this bearer token (default $WASTE_API_KEY)")

    g = ap.add_argument_group("engine")
    g.add_argument("--budget", type=parse_size, default=0, metavar="SIZE",
                   help="hard RAM ceiling, e.g. 48G. 0 lets the engine choose")
    g.add_argument("--ctx", type=bounded_int(0, (1 << 32) - 1),
                   default=0, metavar="N",
                   help="context tokens (0 = container default)")
    g.add_argument("--threads", type=bounded_int(0, (1 << 31) - 1),
                   default=None, metavar="N",
                   help="compute threads (0 = one per core)")
    g.add_argument("--cpus", default=None, metavar="LIST",
                   help="restrict the compute pool to a cpu list, e.g. 0-5 "
                        "or 0-2,6-8; --threads 0 then means one per CPU "
                        "listed. Linux and Windows. Worth it where cores "
                        "differ: on a two-die Ryzen, six threads on one die "
                        "measured 16-25%% faster than six split across both")
    g.add_argument("--cache", choices=sorted(POLICIES), default="lfru")
    g.add_argument("--no-direct-io", action="store_true",
                   help="keep the page cache in the way. The bypass is on "
                        "by default and is what makes the reported hit "
                        "rates the engine's rather than the kernel's")
    g.add_argument("--vision", action="store_true",
                   help="load the vision tower, so requests may carry images")
    g.add_argument("--verify", action="store_true",
                   help="check every expert record's crc32 as it is read; "
                        "for a container copied or downloaded and not read "
                        "since. Costs ~5%% on Kimi-Linear, ~1%% on K3")
    g.add_argument("--usage", default=None, metavar="PATH",
                   help="learned hotlist (default <model>/usage.waste)")
    g.add_argument("--allow-concurrent-open", action="store_true",
                   help="permit another process to load the same container")
    g.add_argument("--performance-profile", choices=PROFILE_NAMES,
                   default=None, metavar="NAME",
                   help="explicit opt-in operating profile; spark-q0 "
                        "requires the request-scoped PM-QoS launcher; "
                        "spark-cuda uses the child-scoped CUDA launcher")

    s = ap.add_argument_group("serving")
    s.add_argument("--max-tokens", type=bounded_int(1, (1 << 32) - 1),
                   default=4096,
                   help="default cap when a request does not set one "
                        "(default 4096). Most clients never set one, and a "
                        "reply that stops at the cap is indistinguishable "
                        "from a model that stopped on its own")
    s.add_argument("--no-thinking", action="store_true",
                   help="answer without the think channel unless a request "
                        "asks for it. K3's reasoning can be most of a reply, "
                        "and at streaming speeds that is a long wait before "
                        "the first word of the answer")
    s.add_argument("--allow-local-images", action="store_true",
                   help="let requests name images by filesystem path. Off by "
                        "default: it lets any client read files the server "
                        "can reach")
    s.add_argument("--prefix-cache", type=parse_size, default=0,
                   metavar="SIZE",
                   help="reserve SIZE inside the RAM budget for exact shared-"
                        "prefix snapshots (off by default; minimum 4K, "
                        "e.g. 2G)")
    s.add_argument("--prefix-cache-entries",
                   type=bounded_int(1, (1 << 31) - 1), default=8,
                   metavar="N",
                   help="hard snapshot entry limit (default 8)")
    s.add_argument("--conversation-head", action="store_true",
                   help="retain one exact mutable conversation checkpoint in "
                        "addition to stable family roots; requires a prefix "
                        "cache and at least two entries")
    s.add_argument("--plan", action="store_true",
                   help="print the memory plan and exit without loading")

    args = ap.parse_args(argv)

    # Reject an impossible retained-cache budget before opening a potentially
    # enormous model. ChatServer repeats this invariant for programmatic
    # callers; the CLI check keeps a typo from paying K3's startup cost first.
    if (args.prefix_cache and
            args.prefix_cache < CONTROLLER_OVERHEAD_BYTES):
        print(f"prefix cache must be 0 (disabled) or at least "
              f"{CONTROLLER_OVERHEAD_BYTES} bytes", file=sys.stderr)
        return 2
    if args.conversation_head and (
            not args.prefix_cache or args.prefix_cache_entries < 2):
        print("--conversation-head requires --prefix-cache and at least two "
              "prefix-cache entries", file=sys.stderr)
        return 2

    model = Path(args.model).expanduser()
    if not model.exists():
        print(f"no such container: {model}", file=sys.stderr)
        return 2
    model_id = args.model_id or model.name.removesuffix(".waste")

    try:
        profile = resolve_profile(
            args.performance_profile, threads=args.threads,
            cpus=args.cpus,
            cache=args.cache, no_direct_io=args.no_direct_io,
            verify=args.verify, environ=os.environ)
    except ProfileError as e:
        print(f"{e}", file=sys.stderr)
        return 2

    request_qos = DisabledRequestQos()
    engine = None
    try:
        if args.plan:
            plan = plan_memory(str(model), args.ctx)
            physical = physical_ram()
            usable = usable_ram()
            ceiling = memory_ceiling()
            print(f"{build_info()}\n")
            print(f"  trunk        {human(plan.trunk_bytes)}")
            print(f"  state        {human(plan.state_bytes)}")
            print(f"  scratch      {human(plan.scratch_bytes)}")
            print(f"  min cache    {human(plan.min_expert_cache)}")
            print(f"  floor        {human(plan.floor_bytes)}")
            if args.prefix_cache:
                print(f"  prefix cache {human(args.prefix_cache)} reserved")
                print(f"  floor+host   {human(plan.floor_bytes + args.prefix_cache)}")
            print(f"  recommended  {human(plan.recommended_bytes)}")
            if plan.vision_bytes:
                print(f"  vision       {human(plan.vision_bytes)} "
                      f"(only with --vision)")
            if physical:
                print(f"\n  host physical RAM           {human(physical)}")
            if usable:
                print(f"  stable process capacity     {human(usable)}")
            if ceiling:
                print(f"  automatic-open ceiling now  {human(ceiling)} (snapshot)")
            return 0

        # Adopt the private control channel before the expensive model load.
        # A strict profile must never start and silently omit its Q0 lease.
        if profile.request_qos_required:
            request_qos = RequestQos.from_env(required=True)
        else:
            # Merely inheriting a descriptor must never enable a machine-wide
            # policy under the default profile.
            discard_control_from_env()
        engine = Engine(
            str(model),
            ram_budget_bytes=args.budget,
            ctx_tokens=args.ctx,
            n_threads=profile.threads,
            cpu_list=profile.cpu_list,
            cache_policy=POLICIES[profile.cache],
            direct_io=profile.direct_io,
            vision=args.vision,
            verify_records=profile.verify_records,
            usage_path=args.usage,
            allow_concurrent_open=args.allow_concurrent_open,
            host_reserved_bytes=args.prefix_cache)
        effective_direct_io = bool(engine.stats()["direct_io"])
        if profile.strict and not effective_direct_io:
            raise ProfileError(
                f"{profile.name} requires effective direct I/O; the model or "
                "filesystem fell back to buffered reads")
        public_profile = profile.public()
        public_profile["engine"]["direct_io_effective"] = effective_direct_io
        engine_stats = engine.stats()
        effective_threads = int(engine_stats["read_ahead_threads"])
        effective_depth = int(engine_stats["read_ahead_depth"])
        storage = public_profile["storage"]
        storage["effective_configuration_reported"] = True
        storage["effective_read_ahead_threads"] = effective_threads
        storage["effective_read_ahead_depth"] = effective_depth
        if profile.strict and (
                effective_threads != storage["requested_read_ahead_threads"] or
                effective_depth != storage["requested_read_ahead_depth"]):
            raise ProfileError(
                f"{profile.name} requires effective read-ahead "
                f"{storage['requested_read_ahead_threads']}/"
                f"{storage['requested_read_ahead_depth']}; engine reported "
                f"{effective_threads}/{effective_depth}")
    except (EngineError, ProfileError, QosError, ValueError) as e:
        request_qos.close()
        if engine is not None:
            engine.close()
        print(f"{e}", file=sys.stderr)
        # Two statuses that say nothing useful on their own when --cpus is
        # what produced them, and here it usually is.
        status = getattr(e, "status", None)
        if args.cpus and status == WASTE_E_ARG:
            print(f"--cpus: not a cpu list: {args.cpus}", file=sys.stderr)
        elif args.cpus and status == WASTE_E_UNSUPPORTED:
            print("--cpus: this platform does not bind threads to CPUs "
                  "(Linux and Windows only)", file=sys.stderr)
        return 1

    try:
        info = engine.model_info()
        used = engine.memory_used()
        print(build_info())
        print(f"model    {model_id} — {info['arch']}, {info['n_layers']} layers, "
              f"{info['n_experts']} experts, ctx {info['ctx_max']}")
        if info.get("quant_summary"):
            print(f"quant    {info['quant_summary']}")
        print(f"memory   {human(used['floor_bytes'])} resident, "
              f"expert cache {human(used['min_expert_cache'])}")
        if args.prefix_cache:
            entry_kind = ("snapshots" if args.conversation_head
                          else "family roots")
            print(f"prefix   {human(args.prefix_cache)} reserved, "
                  f"at most {args.prefix_cache_entries} {entry_kind}")
            if args.conversation_head:
                print("head     one exact mutable conversation checkpoint")
        print(f"thinking {'off by default' if args.no_thinking else 'on'}"
              f" — reasoning_effort per request")
        profile_note = (" — bounded request-scoped Q0"
                        if profile.request_qos_required else "")
        print(f"profile  {profile.name}{profile_note}")
        if args.vision:
            print("vision   on — requests may carry base64 images")
        if not args.api_key and args.host not in ("127.0.0.1", "localhost",
                                                  "::1"):
            print(f"\nWARNING: listening on {args.host} with no --api-key: "
                  f"anyone who can reach this port can use the model.",
                  file=sys.stderr)

        srv = serve(engine, host=args.host, port=args.port, model_id=model_id,
                    api_key=args.api_key,
                    default_max_tokens=args.max_tokens,
                    default_thinking=not args.no_thinking,
                    allow_local_images=args.allow_local_images,
                    prefix_cache_bytes=args.prefix_cache,
                    prefix_cache_entries=args.prefix_cache_entries,
                    conversation_head=args.conversation_head,
                    request_qos=request_qos,
                    performance_profile=public_profile)
    except (EngineError, OSError, QosError, ValueError) as e:
        request_qos.close()
        engine.close()
        print(f"{e}", file=sys.stderr)
        return 1

    shown = args.host if ":" not in args.host else f"[{args.host}]"
    print(f"\nlistening on http://{shown}:{args.port}  "
          f"(POST /v1/chat/completions)")
    # Flush before blocking forever. Redirected to a file, stdout is
    # block-buffered, so without this `python3 -m serve … > log &` shows an
    # empty log for as long as the server runs — which reads exactly like a
    # server that failed to start.
    sys.stdout.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", file=sys.stderr)
    finally:
        srv.shutdown()
        srv.server_close()
        engine.close()
        shutil.rmtree(srv.tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
