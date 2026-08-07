# vhs tape: records examples/demo.gif from the real demo.
# Re-record with: vhs examples/demo.tape   (from the repo root)
Output examples/demo.gif

Set FontSize 13
Set Width 940
Set Height 620
Set TypingSpeed 40ms
Set Padding 12

Type ".venv/bin/python examples/demo.py --steps 400"
Enter
# let prioritized sampling differentiate the priorities
Sleep 11s
# flip to uniform sampling: watch the attention ratio collapse
Type "u"
Sleep 7s
# and back to prioritized: attention recovers
Type "u"
Sleep 7s
Type "q"
Sleep 1s
