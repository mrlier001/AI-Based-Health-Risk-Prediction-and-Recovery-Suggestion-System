c = open("app.py", encoding="utf-8").read()
idx = c.find("Model Performance")
if idx == -1:
    open("check.txt", "w", encoding="utf-8").write("NOT FOUND")
else:
    open("check.txt", "w", encoding="utf-8").write(
        "FOUND at char " + str(idx) + "\n\n" + c[max(0,idx-200):idx+600]
    )
print("done")
