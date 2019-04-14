function generateButtonEvents() {
    var buttons = document.getElementsByTagName("button");
    for (var i = 0; i < buttons.length; i++) {
        buttons[i].onclick = function(e) {
            var buttonText = this.textContent || this.innerText
            var textBox = document.getElementById("text-box")
            var whiteSpaceReg = /\s$/ // Regex to test if string ends with whitespace
            if (whiteSpaceReg.test(textBox.textContent))  {
                textBox.textContent += buttonText + " "
            } else {
                textBox.textContent +=" " + buttonText + " "
            }
        }
    };
}

function createWebsocket() {
    console.log("Starting WebSocket...")
    const socket= new WebSocket('ws://localhost:50007')
    console.log("WebSocket Created...")
    // Connection opened
    socket.addEventListener('open', function (event) {
        console.log("OPEN STARTED")
        socket.send('Hello Server!');
    });

}


// Most likely keep the buttons but just change the textContent of them