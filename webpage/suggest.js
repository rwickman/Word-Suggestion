function onClickSuggestion(e) {
    var buttonText = e.textContent || e.innerText
    var textBox = document.getElementById("text-box")
    var whiteSpaceReg = /\s$/ // Regex to test if string ends with whitespace
    if (whiteSpaceReg.test(textBox.textContent))  {
        textBox.textContent += buttonText + " "
    } else {
        textBox.textContent +=" " + buttonText + " "
    }
    createWebsocket(textBox.textContent)
}

function createWebsocket(text) {
    console.log("Starting WebSocket...")
    const socket= new WebSocket('ws://localhost:50007')
    console.log("WebSocket Created...")
    // Connection opened
    socket.addEventListener('open', function (event) {
        socket.send(text);
       
    });

    socket.addEventListener('message', function (event) {
        var suggestions = JSON.parse(event.data);
        createNewSuggestions(suggestions)
    });
}

function createNewSuggestions(suggestions) {
    var suggestions = suggestions.filter(word => word != "" && word != " " && word != "'" )
    console.log(suggestions)
    
    
    var suggest_con = document.getElementById("suggestions-container")
    // Remove all old suggestions
    while(suggest_con.firstChild) {
        suggest_con.removeChild(suggest_con.firstChild)
    }

   
    for(var i = 0; i < suggestions.length; i++) {
        var div_el = document.createElement("div")
        div_el.setAttribute("id", i)
        div_el.setAttribute("class", "suggestion")
        var button_el = document.createElement("button")
        button_el.setAttribute("type", "button")
        button_el.setAttribute("onclick", "onClickSuggestion(this)")
        button_el.textContent = suggestions[i]
        div_el.appendChild(button_el)
        suggest_con.appendChild(div_el)
    }
}


window.onload = function () {
    // Create the empty suggestions
    createWebsocket("")

    // If a new word is entered instead of button pressed do this
    document.getElementById("text-box").addEventListener("keydown", function(e) {
        console.log("SPACE PRESSED")
        if (e.keyCode == 32) {
            createWebsocket(this.textContent)
        }
    });
}