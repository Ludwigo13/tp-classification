import matplotlib.pyplot as plt
import pandas as pd

def save_dataframe_head_as_png(df, filename):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    
    df = df.head()
    col_widths = [max([len(str(value)) for value in df[col]]) for col in df.columns]
    
    # Normalize column widths to get a scale factor
    width_scale = max(col_widths) / max(10, min(col_widths))
    fig_width = sum(col_widths) / 10  # Adjust overall figure width

    fig, ax = plt.subplots(figsize=(fig_width*0.8, len(df)*0.4))  # Adjust figure size based on DataFrame
    
    # Hide the axes
    ax.axis('off')

    # Render the table on the plot
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', edges='horizontal')
    
    # Adjust table font size
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    
    # Adjust column widths in the table
    for i, col_width in enumerate(col_widths):
        table.auto_set_column_width(i)

    # Save the plot as a PNG image
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
