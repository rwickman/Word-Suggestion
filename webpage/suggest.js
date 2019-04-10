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


// Most likely keep the buttons but just change the textContent of them