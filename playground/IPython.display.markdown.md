---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python id="G0xrM339BM5l"
import pandas as pd
```

```python id="BPeWqz0ZB_vH"
from IPython.display import display, Markdown, Latex, HTML
```

```python tags=[]
df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]}, index=['a', 'b', 'c'])
print(df.to_markdown(index=True))
```

```python tags=[]
display(Latex(df.to_latex(index=True)))
```

```python tags=[]
display(df)
```

```python tags=[]
display(Markdown(df.to_markdown()))
```

```python id="yBOTzL-gEgD5"
df_styled = df.style.set_caption('Test table rendered in LaTeX')
```

```python tags=[]
display(Latex(df_styled.to_latex()))
```

```python
#display(Markdown(df_styled))
df.to_markdown(index=True)
#display(HTML(df_styled.to_html()))
```

<!-- #region id="_2CMWexxb_6I" -->
Table: Das ist die Tabellenüberschrift  

|    |   A |   B |
|:---|----:|----:|
| a  |   1 |   1 |
| a  |   2 |   2 |
| b  |   3 |   3 |
<!-- #endregion -->

```python tags=[]
display(Latex("\\begin{table}\n\\caption{Test table rendered in markdown}\n\\begin{tabular}{lrr}\n{} & {A} & {B} \\\\\na & 1 & 1 \\\\\na & 2 & 2 \\\\\nb & 3 & 3 \\\\\n\\end{tabular}\n\\end{table}\n"))
```

```python
display(Markdown('|    |   A |   B |\n' \
                 '|:---|----:|----:|\n' \
                 '| a  |   1 |   1 |\n' \
                 '| b  |   2 |   2 |\n' \
                 '| c  |   3 |   3 |'))
```

<!-- #region id="0bu4axNPeFOq" -->
Table: File table for all participants

| file_idx|keys |filenames                                             |descriptions               |
|--------:|:----|:-----------------------------------------------------|:--------------------------|
|        1|crit |rdata_all_crit_AHP_edible_Cities_2022-03-18_09-53.csv |criteria (main criteria)   |
|        2|env  |rdata_all_env_AHP_edible_Cities_2022-03-18_09-53.csv  |environmental sub-criteria |
|        3|soc  |rdata_all_soc_AHP_edible_Cities_2022-03-18_09-53.csv  |social sub-criteria        |
|        4|eco  |rdata_all_eco_AHP_edible_Cities_2022-03-18_09-53.csv  |economic sub-criteria      |
<!-- #endregion -->

```python
str_table_caption = 'Test table rendered in Markdown'
str_table_content = df.to_markdown(index=True)

str_table_complete = 'Table: ' + str_table_caption + '\n\n' + str_table_content

print(str_table_complete)
```

```python
display(Markdown(str_table_complete))
```

```python
# Function to render dataframes to markdown table with caption
def func_render_dataframe2Markdown(df, str_caption):
    str_table_complete = 'Table: ' + str_caption + '\n\n' \
                         + df.to_markdown(index=True)
    display(Markdown(str_table_complete))
```

```python
str_table_caption = 'Test table rendered in Markdown'

func_render_dataframe2Markdown(df, str_table_caption)
```

```python

```
