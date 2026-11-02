"""Headless checks for GUI startup and stage navigation contracts.

Linux CI's [dev] extra intentionally does not include PySide6. These
checks exercise the production navigation method against an adversarial
stack with scrambled numeric indices, while the Qt tests exercise full UI.
"""
from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] /
          "src/scorpio_pipe/ui/launcher_window.py")


def _method(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and
               x.name == "LauncherWindow")
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and
              x.name == name)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],
                                                    type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    return namespace[name]


class _UnorderedStack:
    """Actual widget position intentionally unrelated to stage page index."""

    def __init__(self):
        self.items = [object() for _ in range(12)]
        self.current = self.items[0]

    def indexOf(self, item):
        try:
            return self.items.index(item)
        except ValueError:
            return -1

    def currentIndex(self):
        return self.indexOf(self.current)

    def setCurrentWidget(self, item):
        assert item in self.items
        self.current = item

    def setCurrentIndex(self, index):
        # Old production behavior would select the wrong item here.
        self.current = self.items[index]


class _Fake:
    _on_step_changed = _method("_on_step_changed")

    def __init__(self):
        self.stack = _UnorderedStack()
        self.stack.items = list(reversed(self.stack.items))
        self._stage_page_index = [0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11]
        self._page_widgets = {i: widget for i, widget in enumerate(
            list(reversed(self.stack.items)))}
        self.builders = []
        self.logs = []
        self.status_calls = 0

    def _ensure_page_built(self, i):
        self.builders.append(i)

    def _log_exception(self, exc):
        self.logs.append(str(exc))

    def _refresh_statusbar(self):
        self.status_calls += 1

    def _sync_outputs_action_state(self):
        pass

    def _update_outputs_tool_context(self):
        pass

    def _update_wavesol_stepper(self):
        pass


def test_production_step_navigation_uses_widget_identity_not_index():
    window = _Fake()
    for row in (0, 4, 3, 2, 1, 11, 0):
        index = window._stage_page_index[row]
        window._on_step_changed(row)
        assert window.stack.current is window._page_widgets[index]
    assert window.builders == [0, 3, 4, 2, 1, 11, 0]
    assert not window.logs
    assert window.status_calls == 7


def test_first_project_page_is_built_synchronously_before_first_show():
    code = SOURCE.read_text(encoding="utf-8")
    cls = next(x for x in ast.parse(code).body if isinstance(x, ast.ClassDef)
               and x.name == "LauncherWindow")
    ctor = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                and x.name == "__init__")
    # Validate the final initialization block, not an incidental helper.
    statements = [ast.unparse(n) for n in ctor.body]
    assert any("self._ensure_page_built(0)" in x for x in statements)
    assert any("self.stack.setCurrentWidget(self.page_project)" in x
               for x in statements)
    assert not any("QTimer.singleShot(0" in x and "_ensure_page_built" in x
                   for x in statements)


def test_production_inspect_population_is_deferred_until_setup_exists():
    class ProjectOnly:
        _populate_objects_from_inspect = _method("_populate_objects_from_inspect")
        _pages_built = {0}
        _inspect = object()

    project = ProjectOnly()
    # No combo_object field is intentional; this exactly reproduces
    # the Windows AttributeError after inspecting the 306-frame BTA night.
    project._populate_objects_from_inspect()
    assert not hasattr(project, "combo_object")


def test_production_lazy_setup_construction_replays_finished_inspect():
    from types import SimpleNamespace

    class PlaceholderStack:
        def __init__(self, old):
            self.items = [object(), old]
            self.active = old

        def currentWidget(self):
            return self.active

        def removeWidget(self, widget):
            self.items.remove(widget)
            if self.active is widget:
                self.active = self.items[0]

        def insertWidget(self, index, widget):
            self.items.insert(index, widget)

        def setCurrentWidget(self, widget):
            assert widget in self.items
            self.active = widget

    class SetupAfterInspect:
        _ensure_page_built = _method("_ensure_page_built")

        def __init__(self):
            old = object()
            self.new_setup = object()
            self._pages_built = {0}
            self._page_specs = {
                1: ("setup", "page_config", lambda: self.new_setup),
            }
            self._page_widgets = {1: old}
            self.stack = PlaceholderStack(old)
            self._stage_page_index = [0, 1]
            self.steps = SimpleNamespace(currentRow=lambda: 1)
            self._inspect = SimpleNamespace(objects=["N2146sat"])
            self.populated = []
            self.errors = []
            self.enable_updates = 0

        def _populate_objects_from_inspect(self):
            self.populated.append(tuple(self._inspect.objects))

        def _log_exception(self, exc):
            self.errors.append(str(exc))

        def _update_enables(self):
            self.enable_updates += 1

    win = SetupAfterInspect()
    win._ensure_page_built(1)
    assert win.populated == [("N2146sat",)]
    assert win._pages_built == {0, 1}
    assert win.stack.currentWidget() is win.new_setup
    assert win.page_config is win.new_setup
    assert win.enable_updates == 1
    assert not win.errors
