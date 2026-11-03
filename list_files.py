from pathlib import Path
base = Path(r'c:\Users\DELL\Downloads\assignment1')
for p in base.iterdir():
    print(repr(str(p.name)))
    print(p.name)
