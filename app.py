from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Franka Robot VLM - Hugo Markoff</title>
      <style>
        body {
          font-family: Arial, sans-serif;activate
          background: #f0f2f5;
          display: flex;
          justify-content: center;
          align-items: center;
          height: 100vh;
          margin: 0;
        }
        #container {
          background: #fff;
          border-radius: 10px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.15);
          width: 800px;
          max-width: 95%;
          padding: 20px;
        }
        #videoContainer {
          position: relative;
          width: 100%;
        }
        video {
          width: 100%;
          border-radius: 10px;
          background: #000;
        }
        #cameraSelect {
          margin-top: 10px;
          width: 100%;
          padding: 8px;
          border: 1px solid #ccc;
          border-radius: 4px;
        }
        #chatContainer {
          margin-top: 20px;
          border: 1px solid #ddd;
          border-radius: 10px;
          height: 300px;
          display: flex;
          flex-direction: column;
        }
        #chatMessages {
          flex: 1;
          padding: 10px;
          overflow-y: auto;
          border-bottom: 1px solid #ddd;
        }
        #chatInputContainer {
          display: flex;
        }
        #chatInput {
          flex: 1;
          padding: 10px;
          border: none;
          outline: none;
          border-radius: 0 0 0 10px;
          font-size: 16px;
        }
        #sendButton {
          padding: 10px 20px;
          background-color: #007BFF;
          color: #fff;
          border: none;
          cursor: pointer;
          border-radius: 0 0 10px 0;
          font-size: 16px;
        }
        #sendButton:hover {
          background-color: #0056b3;
        }
      </style>
    </head>
    <body>
      <div id="container">
        <div id="videoContainer">
          <video id="video" autoplay playsinline></video>
          <select id="cameraSelect"></select>
        </div>
        <div id="chatContainer">
          <div id="chatMessages"></div>
          <div id="chatInputContainer">
            <input type="text" id="chatInput" placeholder="Type your message..." />
            <button id="sendButton">Send</button>
          </div>
        </div>
      </div>
    
      <script>
        // Populate available video input devices
        async function getCameraOptions() {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const videoDevices = devices.filter(device => device.kind === 'videoinput');
          const cameraSelect = document.getElementById('cameraSelect');
          cameraSelect.innerHTML = '';
          videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || 'Camera ' + (index + 1);
            cameraSelect.appendChild(option);
          });
        }
    
        // Start the video stream with a given device ID
        async function startStream(deviceId) {
          const video = document.getElementById('video');
          if (window.stream) {
            window.stream.getTracks().forEach(track => track.stop());
          }
          const constraints = {
            video: { deviceId: deviceId ? { exact: deviceId } : undefined }
          };
          try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            window.stream = stream;
            video.srcObject = stream;
          } catch (error) {
            console.error('Error accessing camera:', error);
          }
        }
    
        // When the camera dropdown changes, restart the stream
        document.getElementById('cameraSelect').addEventListener('change', function() {
          startStream(this.value);
        });
    
        // Initialize camera options and start default stream on load
        window.addEventListener('load', async () => {
          await getCameraOptions();
          const cameraSelect = document.getElementById('cameraSelect');
          if(cameraSelect.options.length > 0) {
            startStream(cameraSelect.options[0].value);
          }
        });
    
        // Chat functionality: add message to chat window
        document.getElementById('sendButton').addEventListener('click', function() {
          const chatInput = document.getElementById('chatInput');
          const message = chatInput.value.trim();
          if (message !== '') {
            addChatMessage('You', message);
            chatInput.value = '';
          }
        });
    
        // Append message to chat area
        function addChatMessage(sender, message) {
          const chatMessages = document.getElementById('chatMessages');
          const messageDiv = document.createElement('div');
          messageDiv.style.marginBottom = '10px';
          messageDiv.innerHTML = '<strong>' + sender + ':</strong> ' + message;
          chatMessages.appendChild(messageDiv);
          chatMessages.scrollTop = chatMessages.scrollHeight;
        }
      </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(debug=True)
