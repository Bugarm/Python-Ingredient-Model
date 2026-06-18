from ultralytics import YOLO


# Export the trained Ultralytics checkpoint directly to ONNX.
model = YOLO('best.pt')
model.export(format='onnx', opset=12)

print("Model converted to ONNX successfully!")