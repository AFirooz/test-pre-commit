# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,scripts//py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
base = "H3N3	H3N3F1	H3N3S1	H3N4"
base = base.split("\t")

query = "H4N2	H3N3	H5N2"
query = query.split("\t")

print(base[0])
print(query[0])

# type(query[0])
