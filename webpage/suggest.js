function onClickSuggestion(e) {
    var buttonText = e.textContent || e.innerText
    var textBox = document.getElementById("text-box")
    var whiteSpaceReg = /\s$/ // Regex to test if string ends with whitespace
    if (whiteSpaceReg.test(textBox.textContent))  {
        textBox.textContent += buttonText + " "
    } else {
        textBox.textContent +=" " + buttonText + " "
    }
    createWebsocket(JSON.stringify({"class" : e.className , "text" : textBox.textContent}))
}

function createWebsocket(text) {
    console.log("Starting WebSocket...")
    const socket= new WebSocket('ws://localhost:50007')
    console.log("WebSocket Created...")
    // Connection opened
    socket.addEventListener('open', function (event) {
        console.log(text)
        socket.send(text);
    });

    socket.addEventListener('message', function (event) {
        var suggestions = JSON.parse(event.data);
        createNewSuggestions(suggestions)
    });
}

function createNewSuggestions(suggestions) {
    console.log(typeof suggestions["lsh suggestions"])
    var lsh_suggestions = suggestions["lsh suggestions"]
    
    var suggestions = suggestions["suggestions"].filter(word => word != "" && word != " " && word != "'" )
    if (lsh_suggestions["movie predictions"]) {
        var lsh_suggestions_movies = lsh_suggestions["movie predictions"].filter(word => word != "" && word != " " && word != "'" )
    } else {
        var lsh_suggestions_movies = []
    }
    if (lsh_suggestions["lyric predictions"]) {
        var lsh_suggestions_lyrics = lsh_suggestions["lyric predictions"].filter(word => word != "" && word != " " && word != "'" )
    } else {
        var lsh_suggestions_movies = []
    }
    console.log(lsh_suggestions)
    
    
    var suggest_con = document.getElementById("suggestions-container")
    // Remove all old suggestions
    while(suggest_con.firstChild) {
        suggest_con.removeChild(suggest_con.firstChild)
    }
    var id_num = 0
    // Add the new suggestions
    for(var i = 0; i < suggestions.length; i++) {
        var div_el = document.createElement("div")
        div_el.setAttribute("id", id_num++)
        div_el.setAttribute("class", "suggestion")
        var button_el = document.createElement("button")
        button_el.setAttribute("type", "button")
        button_el.setAttribute("onclick", "onClickSuggestion(this)")
        button_el.textContent = suggestions[i]
        div_el.appendChild(button_el)
        suggest_con.appendChild(div_el)
    }

    var lsh_suggest_con = document.getElementById("lsh-suggestions-container")
    // Remove all old suggestions
    while(lsh_suggest_con.firstChild) {
        lsh_suggest_con.removeChild(lsh_suggest_con.firstChild)
    }

    // Add the new LSH RNN suggestions
    for(var i = 0; i < lsh_suggestions_movies.length; i++) {
        var div_el = document.createElement("div")
        div_el.setAttribute("id", id_num++)
        div_el.setAttribute("class", "suggestion")
        var button_el = document.createElement("button")
        button_el.setAttribute("type", "button")
        button_el.setAttribute("onclick", "onClickSuggestion(this)")
        button_el.setAttribute("class", "movies")
        button_el.textContent = lsh_suggestions_movies[i]
        div_el.appendChild(button_el)
        lsh_suggest_con.appendChild(div_el)
    }

    // Add the new LSH RNN suggestions
    for(var i = 0; i < lsh_suggestions_lyrics.length; i++) {
        var div_el = document.createElement("div")
        div_el.setAttribute("id", id_num++)
        div_el.setAttribute("class", "suggestion")
        var button_el = document.createElement("button")
        button_el.setAttribute("type", "button")
        button_el.setAttribute("onclick", "onClickSuggestion(this)")
        button_el.setAttribute("class", "lyrics")
        button_el.textContent = lsh_suggestions_lyrics[i]
        div_el.appendChild(button_el)
        lsh_suggest_con.appendChild(div_el)
    }
}

window.onload = function () {
    // Create the empty suggestions
    createWebsocket(JSON.stringify({"class" : "", "text" : ""}))

    // If a new word is entered instead of button pressed do this
    document.getElementById("text-box").addEventListener("keydown", function(e) {
        console.log("SPACE PRESSED")
        if (e.keyCode == 32) {
            createWebsocket(JSON.stringify({"class" : this.className , "text" : this.textContent}))
        }
    });
}