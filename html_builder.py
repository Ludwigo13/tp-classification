import json

# Function to generate HTML for a collapsible dictionary
def generate_html(data, indent=0):
    html = ''
    for key, value in data.items():
        if isinstance(value, dict):  # Always make dictionaries collapsible
            html += f'''
            <details style="margin-left:{indent}em;">
                <summary><strong>{key}</strong></summary>
                {generate_html(value, indent + 1)}
            </details>
            '''
        elif isinstance(value, str) and value.endswith('.png'):
            html += f'''
            <div class="key-value" style="margin-left:{indent}em;">
                <strong>{key}:</strong><br>
                <img src="{value}" alt="{key}" style="max-width: 100%; height: auto; border-radius: 5px;"/>
            </div>
            '''
        else:
            # Display key-value pairs normally        
            html += f'<div class="key-value" style="margin-left:{indent}em;"><strong>{key}:</strong> {value}</div>'
    return html

# Function to wrap the content in HTML structure
def wrap_in_html(body_content, title):
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                background-color: #f9f9f9;
                color: #333;
                margin: 0;
                padding: 20px;
            }}
            h1 {{
                text-align: center;
                color: #444;
            }}
            details {{
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                padding: 10px;
                transition: all 0.3s ease;
            }}
            summary {{
                font-size: 1.2em;
                font-weight: 600;
                cursor: pointer;
                color: #007BFF;
                outline: none;
                user-select: none;
            }}
            summary::marker {{
                color: #007BFF;
            }}
            details[open] {{
                border-left: 4px solid #007BFF;
            }}
            .key-value {{
                padding: 5px;
                background-color: #f5f5f5;
                border-radius: 4px;
                margin: 5px 0;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }}
            .key-value strong {{
                color: #555;
            }}
            details > div {{
                margin-left: 15px;
            }}
            @media (max-width: 600px) {{
                body {{
                    padding: 10px;
                }}
                summary {{
                    font-size: 1em;
                }}
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        {body_content}
    </body>
    </html>
    '''

# Load JSON from a file
def generate_collapsible_html(json_file, html_file):
    data = None
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Generate collapsible HTML
    body_content = generate_html(data)

    # Wrap it in full HTML structure
    full_html = wrap_in_html(body_content, "Models Report")

    # Output the HTML to a file
    with open(html_file, 'w') as f:
        f.write(full_html)

    print(f"HTML file generated: {html_file}")

# Example usage
if __name__ == "__main__":
    json_file_path = 'docs/report.json'
    html_file_path = 'docs/index.html'
    generate_collapsible_html(json_file_path, html_file_path)