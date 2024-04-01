function expand(button) {
    id = button.id
    target = document.querySelector(`#div${id}`)
    arrow = document.querySelector(`#arrow${id}`)
    if (target.style.display == "none") {
        target.style.display = "inline"
    } else {
        target.style.display = "none"
    }

    if (arrow.style.transform == "scaleY(1)") {

        arrow.style.transform = "scaleY(-1)"
    } else {

        arrow.style.transform = "scaleY(1)"
    }
}