import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Image_Conversion import *
from image_upload import process_image_pipeline
from qa import answer_question

def main(img_path, output_image_path, resolution, question):
    """Runs the complete pipeline from image processing to question answering."""
    try:
        json_file_path = process_image_pipeline(img_path, output_image_path, resolution)
        
        if "error" in json_file_path:
            print("Error in processing image pipeline.")
            return json_file_path
        
        with open(json_file_path, "r") as file:
            ocr_data = json.load(file)
        
        context = " ".join([entry[col]["text"] for entry in ocr_data for col in entry])
        
        answer = answer_question(context, question)
        print(f"Answer: {answer}")
        return answer
    
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        return {"error": str(e)}

# Example execution
if __name__ == "__main__":
    img_path = ".input/sample_input_1.png"
    output_image_path = ".output/sample_output_1.png"
    resolution = (2000, 2000)
    question = "What is the size of the covid-19 Train data?"
    main(img_path, output_image_path, resolution, question)