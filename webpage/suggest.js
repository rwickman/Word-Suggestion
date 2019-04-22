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

function createNewSuggestions(suggestions_d) {
    var suggestions = suggestions_d["suggestions"].filter(word => word != "" && word != " " && word != "'" )
    
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

    var lsh_suggestions = suggestions_d["lsh suggestions"]
    var h3_similar = document.getElementById("similar-user-suggestions")
    var lsh_suggest_con = document.getElementById("lsh-suggestions-container")
    
    // Remove all old LSH suggestions
    while(lsh_suggest_con.firstChild) {
        lsh_suggest_con.removeChild(lsh_suggest_con.firstChild)
    }
    
    if (lsh_suggestions && ( (lsh_suggestions["movie predictions"] && lsh_suggestions["movie predictions"].length > 0) || (lsh_suggestions["lyric predictions"] && lsh_suggestions["lyric predictions"].length > 0) )) {
        console.log(lsh_suggestions)
        if (lsh_suggestions["movie predictions"]) 
            console.log("TRUEEE")
        h3_similar.setAttribute("class", "Suggested-Words-Text")
        if (lsh_suggestions["movie predictions"] && lsh_suggestions["movie predictions"].length > 0) {
            var lsh_suggestions_movies = lsh_suggestions["movie predictions"].filter(word => word != "" && word != " " && word != "'" )
        } else {
            var lsh_suggestions_movies = []
        }
        if (lsh_suggestions["lyric predictions"] && lsh_suggestions["lyric predictions"].length > 0) {
            var lsh_suggestions_lyrics = lsh_suggestions["lyric predictions"].filter(word => word != "" && word != " " && word != "'" )
        } else {
            var lsh_suggestions_lyrics = []
        }
        

        // Add the new LSH RNN suggestions
        var flex_con_1 = document.createElement("div")
        flex_con_1.setAttribute("class", "flex-container")
        for(var i = 0; i < lsh_suggestions_movies.length; i++) {
            if (i == 0) {
                var h2_el = document.createElement("h2")
                h2_el.textContent = "User 1(quotes): "
                flex_con_1.appendChild(h2_el)
                lsh_suggest_con.appendChild(flex_con_1)
            }
            var div_el = document.createElement("div")
            div_el.setAttribute("id", id_num++)
            div_el.setAttribute("class", "suggestion")
            var button_el = document.createElement("button")
            button_el.setAttribute("type", "button")
            button_el.setAttribute("onclick", "onClickSuggestion(this)")
            button_el.setAttribute("class", "movies")
            button_el.textContent = lsh_suggestions_movies[i]
            div_el.appendChild(button_el)
            flex_con_1.appendChild(div_el)
        }

        // Add the new LSH RNN suggestions
        var flex_con_2 = document.createElement("div")
        flex_con_2.setAttribute("class", "flex-container")
        for(var i = 0; i < lsh_suggestions_lyrics.length; i++) {
            if (i == 0) {
                var h2_el = document.createElement("h2")
                h2_el.textContent = "User 2(lyrics): "
                flex_con_2.appendChild(h2_el)
                lsh_suggest_con.appendChild(flex_con_2)
            }
            var div_el = document.createElement("div")
            div_el.setAttribute("id", id_num++)
            div_el.setAttribute("class", "suggestion")
            var button_el = document.createElement("button")
            button_el.setAttribute("type", "button")
            button_el.setAttribute("onclick", "onClickSuggestion(this)")
            button_el.setAttribute("class", "lyrics")
            button_el.textContent = lsh_suggestions_lyrics[i]
            div_el.appendChild(button_el)
            flex_con_2.appendChild(div_el)
        }
    } else {
        h3_similar.setAttribute("class", "Suggested-Words-Text hide")
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