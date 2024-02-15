import ocr_tesseract
import echo_server
import echo_client

echo_server.server()
print("receive sucess 'address.jpg'")

ocr_tesseract.ocr()
print("'ocr_result.txt' was created")

echo_client.client()
print("send success 'ocr_result.txt'")