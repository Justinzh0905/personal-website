window.onload = main

function main() {
    const book = document.getElementById('bookImg')
    const isbn = book.getAttribute('src')
    fetch("https://www.googleapis.com/books/v1/volumes?q=isbn:" + isbn)
    .then(response => response.json())
    .then(data => data.items[0]['volumeInfo'])
    .then(info => {
        book.src = info["imageLinks"]["thumbnail"]
        author = info['authors'][0]
        title = info['title']
        const details = document.getElementById('bookBlurb')
        details.innerHTML = `Title: ${title} <br> Author: ${author} <br> `
        console.log(info)
    })
}
