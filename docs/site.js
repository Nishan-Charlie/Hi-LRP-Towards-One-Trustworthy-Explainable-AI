// Copy buttons. A <pre data-copy> gets one automatically; a <button class="copy"
// data-target="id"> copies the element it names (the BibTeX block).
(function () {
  function copyText(text, button, idle) {
    function done(label) {
      button.textContent = label;
      setTimeout(function () { button.textContent = idle; }, 2000);
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(
        function () { done("Copied"); },
        function () { done("Copy failed: select the text instead"); });
    } else {
      var area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      try { document.execCommand("copy"); done("Copied"); }
      catch (e) { done("Copy failed: select the text instead"); }
      area.remove();
    }
  }

  document.querySelectorAll("button.copy[data-target]").forEach(function (button) {
    var idle = button.textContent;
    button.setAttribute("aria-live", "polite");
    button.addEventListener("click", function () {
      copyText(document.getElementById(button.dataset.target).innerText, button, idle);
    });
  });

  document.querySelectorAll("pre[data-copy]").forEach(function (pre) {
    var holder = document.createElement("div");
    holder.className = "snippet";
    pre.parentNode.insertBefore(holder, pre);
    holder.appendChild(pre);
    var button = document.createElement("button");
    button.type = "button";
    button.className = "copy";
    button.textContent = "Copy";
    button.setAttribute("aria-live", "polite");
    button.addEventListener("click", function () { copyText(pre.innerText, button, "Copy"); });
    holder.appendChild(button);
  });
})();
