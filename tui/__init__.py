"""Terminal UI for the EoSS binary modifier (curses port of VisualModifier).

Launch with ``eoss tui <binary> [--sm N | --json PATH]``.

Modules:

* :mod:`eoss.tui.decode` -- pure instruction decoding / rendering logic
* :mod:`eoss.tui.elfview` -- full ELF image parsing (.nv.info, notes, symbols)
* :mod:`eoss.tui.model` -- editor state: edits, undo, save
* :mod:`eoss.tui.app` -- the curses application (design "3a")
"""
