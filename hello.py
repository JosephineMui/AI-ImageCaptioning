import gradio as gr

'''
This code defines a simple function `greet` that takes a name as input and returns a 
greeting message. The Gradio interface is created using the `gr.Interface` class, 
where the function is specified along with the input and output types. Finally, 
the interface is launched on the specified server and port.

To run: python hello.py
Then open http://localhost:7860 in your web browser to see the interface and test 
the greeting function.
'''
def greet(name):
    return f"Hello, {name}!"

# Create the Gradio interface
iface = gr.Interface(fn=greet, inputs="text", outputs="text")

iface.launch(server_name="0.0.0.0", server_port= 7860)

# iface.launch()