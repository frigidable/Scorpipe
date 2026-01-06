"""PyInstaller entry-point for Scorpipe GUI with console (for debugging)."""

import sys
import traceback
from pathlib import Path


def main() -> None:
    try:
        from scorpio_pipe.ui.launcher_app import main as gui_main
        gui_main()
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"\n{'=' * 80}", file=sys.stderr)
        print(f"CRITICAL ERROR in Scorpio Pipe GUI", file=sys.stderr)
        print(f"{'=' * 80}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

        log_dir = Path.home() / ".scorpipe_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "gui_crash.log"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(traceback.format_exc() + "\n")
            print(f"\nLog saved to: {log_file}", file=sys.stderr)
        except Exception:
            pass

        print(f"\nPress Enter to exit...", file=sys.stderr)
        input()
        sys.exit(1)


if __name__ == "__main__":
    main()