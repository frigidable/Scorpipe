"""PyInstaller entry-point for Scorpipe GUI with console output (debugging version).

This creates scorpipe-debug.exe with a console window, so users can see tracebacks
and error messages when something goes wrong.

Useful for troubleshooting installation/startup issues.
"""

import sys
import traceback
from pathlib import Path


def main() -> None:
    """Launch the GUI with console output for debugging."""
    try:
        from scorpio_pipe.ui.launcher_app import main as gui_main
        gui_main()
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 0)
    except Exception as e:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"CRITICAL ERROR in Scorpio Pipe GUI", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)

        # Try to save to log file
        log_dir = Path.home() / ".scorpipe_logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "gui_crash.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"DEBUG BUILD CRASH:\n")
                f.write(traceback.format_exc())
                f.write(f"\n{'='*80}\n")
            print(f"\nLog saved to: {log_file}", file=sys.stderr)
        except Exception:
            pass

        print(f"\nPress Enter to exit...", file=sys.stderr)
        try:
            input()
        except EOFError:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()