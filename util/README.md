
# Sentiment Analysis Presentation Generator

This project contains a single Python program that generates a PowerPoint presentation for the "Sentiment Analysis on Twitter Data" project. In this program, both the presentation content (data) and the code are included in the same file but are organized into separate sections for clarity.

## File Structure

```
.
└── presentation_generator.py   # Single file with both data and code
└── README.md                   # This README file
```

## Requirements

- Python 3.x
- [python-pptx](https://python-pptx.readthedocs.io/en/latest/)

## Installation

Install the required package using pip:

```bash
pip install python-pptx
```

## Program Overview

The program file `presentation_generator.py` includes two main sections:

1. **Presentation Data Section:**  
   Contains a data structure (a list of dictionaries) that holds all the slide titles and corresponding content. This section makes it easy to update slide details without modifying the actual slide generation code.

2. **Presentation Generation Code Section:**  
   Contains the logic for generating the PowerPoint presentation using the `python-pptx` library. It reads the data defined in the previous section and creates each slide accordingly.

## Usage

1. **Review or Modify Data:**
   - Open the `presentation_generator.py` file.
   - In the “Presentation Data” section (typically near the top), you will find a list named `presentation_data` where each slide’s title and content are defined.
   - You can update the slide content directly in this section if needed.

2. **Run the Program:**
   - Open a terminal in the directory containing `presentation_generator.py`.
   - Run the script using:
     ```bash
     python presentation_generator.py
     ```
   - The program will generate a PowerPoint file named `Sentiment_Analysis_on_Twitter_Data.pptx` in your working directory.

3. **Manual Customization:**
   - Open the generated `.pptx` file in PowerPoint (or a similar tool).
   - Add images or adjust formatting where needed (e.g., for slides indicating image placeholders like flowchart, histogram, or graphs).

## Example Program Structure

The single file program (`presentation_generator.py`) is organized similar to this:

```python
# =====================
# Presentation Data Section
# =====================

presentation_data = [
    {
        "title": "Sentiment Analysis on Twitter Data",
        "content": ("IE7500 – Advanced Topics in Machine Learning\n"
                    "Subash Ramanathan, Ahantya Vempati, Yi Ren\n"
                    "April 2025\n"
                    "Code Repository: https://github.com/Northeastern-MSDAE/IE7500")
    },
    {
        "title": "Overview & Agenda",
        "content": ("Agenda:\n"
                    "• Introduction & Motivation\n"
                    "• Research & Influences\n"
                    "• Methodology & Implementation\n"
                    "• Results, Comparative Analysis & Error Analysis\n"
                    "• Conclusion & Future Work\n"
                    "• Q&A")
    },
    # ... more slide data ...
    {
        "title": "Q&A and Discussion",
        "content": "Questions & Answers"
    }
]

# =====================
# Presentation Generation Code Section
# =====================

from pptx import Presentation
from pptx.util import Inches, Pt

def create_presentation(data, output_filename="Sentiment_Analysis_on_Twitter_Data.pptx"):
    prs = Presentation()
    slide_layout_title = prs.slide_layouts[0]
    slide_layout_content = prs.slide_layouts[1]

    for i, slide_data in enumerate(data):
        if i == 0:
            slide = prs.slides.add_slide(slide_layout_title)
            slide.shapes.title.text = slide_data["title"]
            slide.placeholders[1].text = slide_data["content"]
        else:
            slide = prs.slides.add_slide(slide_layout_content)
            slide.shapes.title.text = slide_data["title"]
            slide.shapes.placeholders[1].text = slide_data["content"]

    prs.save(output_filename)
    print(f"Presentation created: {output_filename}")

if __name__ == "__main__":
    create_presentation(presentation_data)
```

## Customization

- **Updating Slide Content:**  
  Modify the `presentation_data` list to adjust slide titles or contents.
  
- **Enhancements:**  
  You can extend the code section to include image insertion or style modifications based on your requirements.


