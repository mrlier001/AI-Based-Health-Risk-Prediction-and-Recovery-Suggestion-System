lines = open("app.py", encoding="utf-8").readlines()
result = []
in_form = False

for i, line in enumerate(lines):
    lineno = i + 1
    stripped = line.lstrip()
    spaces = len(line) - len(stripped)

    # Line 385 is: "        with st.form("health_form"):"
    # Everything from line 386 to 452 is the form body at wrong 8-space indent
    # It should be at 12-space indent (inside expander + form)
    if lineno == 385:
        in_form = True
        result.append(line)
        continue

    # Line 453 is the closing of the expander, back to 4-space indent
    if in_form and lineno >= 453:
        in_form = False

    if in_form:
        if stripped == "" or stripped == "\n":
            result.append("\n")
        elif spaces == 8:
            # Add 4 more spaces
            result.append("            " + stripped)
        else:
            result.append(line)
    else:
        result.append(line)

open("app.py", "w", encoding="utf-8").writelines(result)
print("done")
