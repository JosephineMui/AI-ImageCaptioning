'''
Hugging Face is widely known for its open-source library called "Transformers" which 
provides thousands of pre-trained models to the community. The library supports a wide 
range of NLP tasks, such as translation, summarization, text generation, and more. 
Transformers has contributed significantly to the recent advancements in NLP, as it has 
made state-of-the-art models, such as BERT, GPT-2, and GPT-3, accessible to researchers 
and developers worldwide.

Transformers library includes a model that can be used to capture information from images. 
The BLIP, or Bootstrapping Language-Image Pre-training, model is a tool that helps computers 
understand and generate language based on images.

"AutoProcessor" and "BlipForConditionalGeneration" are components of the BLIP model, 
which is a vision-language model available in the Hugging Face Transformers library.

AutoProcessor : This is a processor class that is used for preprocessing data for the BLIP model. 
It wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor. This means it 
can handle both image and text data, preparing it for input into the BLIP model.

    Note: A tokenizer is a tool in natural language processing that breaks down text into smaller, 
    manageable units (tokens), such as words or phrases, enabling models to analyze and understand 
    the text.

BlipForConditionalGeneration : This is a model class that is used for conditional text generation 
given an image and an optional text prompt. In other words, it can generate text based on an input 
image and an optional piece of text. This makes it useful for tasks like image captioning or visual 
question answering, where the model needs to generate text that describes an image or answer a 
question about an image.

'''
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the BLIP model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image from a URL
image_path = "TwinCats.png"  # Replace with your image path
# Open the image and convert it to RGB format
# The PIL library is used to open the image file and convert it to RGB format, 
# which is a common color mode for images.
'''
Why this is useful:

It standardizes image format so the model gets consistent input.
It avoids issues with images in modes like RGBA, L (grayscale), or P (palette), 
which can break or reduce quality in preprocessing pipelines.
'''
image = Image.open(image_path).convert('RGB')

# Prepare the inputs for the model
# The processor takes the image and an optional text prompt (in this case, "an image of") 
# and processes them into a format suitable for the model.

# The "return_tensors='pt'" argument specifies that the output should be in the form of PyTorch tensors, 
# which are the data structures used by the PyTorch deep learning framework. This allows the model to 
# efficiently process the input data.
inputs = processor(image, "the image of", return_tensors="pt")

# Generate the caption for the image
# The model generates a caption based on the processed inputs. The "generate" method is used
# to produce the output text, which is the caption describing the image.

'''
The two asterisks (**) in Python are used in function calls to unpack dictionaries and pass items 
in the dictionary as keyword arguments to the function. **inputs is unpacking the inputs dictionary 
and passing its items as arguments to the model.

The argument max_length=50 specifies that the model should generate a caption of up to 50 tokens in length.
'''
out = model.generate(**inputs, max_length=50)

# Decode the generated caption
# The processor's decode method is used to convert the generated output from the model (which is in the 
# form of token IDs) back into human-readable text. The skip_special_tokens=True argument tells the 
# decoder to ignore any special tokens that may be present in the output, such as padding or 
# end-of-sequence tokens.
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)

