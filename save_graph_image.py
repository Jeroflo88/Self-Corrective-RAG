from main import app

def save_graph():
    print("Generating graph image...")
    try:
        # Get the graph image as PNG bytes
        png_bytes = app.get_graph().draw_mermaid_png()
        
        # Save to file
        output_file = "workflow.png"
        with open(output_file, "wb") as f:
            f.write(png_bytes)
        
        print(f"Graph saved to {output_file}")
    except Exception as e:
        print(f"Error saving graph: {e}")
        print("Make sure you have the necessary dependencies installed (e.g., grandalf or similar for layout if required by your version, though mermaid usually works via API or local).")

if __name__ == "__main__":
    save_graph()
