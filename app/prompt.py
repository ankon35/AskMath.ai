# prompt.py
def custom_prompt(image_content):
    """
    Custom prompt to guide Gemini's model in extracting math content.

    :param image_content: Extracted content from the image.
    :return: Refined text for math content.
    """
    prompt = f"""
    Extract the mathematical equations and text from the image below.
    Focus on the algebraic expressions, symbols, fractions, integrals, and equations in the image.
    Make sure to handle both printed and handwritten math content accurately.

    The image content is: {image_content}

    Provide me the clean, readable math expressions and equations from the image.
    Format the output exactly as follows:

    "The text extracted from the image is:

    [Extracted Text/Equation in LaTeX format]"
    """
    return prompt