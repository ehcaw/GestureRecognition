function write_letter() {
  fetch("/get_label")
    .then((response) => response.text())
    .then((data) => {
      const textBlock = document.getElementById("frame");
      textBlock.textContent = textBlock.textContent + data.frame;
      textBlock.style.fontSize = `${data.frame.length}px`;
    });
}

setInterval(write_letter, 500);
