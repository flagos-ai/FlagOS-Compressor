document.addEventListener("DOMContentLoaded", () => {
  const primarySidebar = document.getElementById("pst-primary-sidebar");

  document.querySelectorAll(".primary-toggle").forEach((toggle) => {
    toggle.addEventListener(
      "click",
      (event) => {
        if (!window.matchMedia("(min-width: 992px)").matches || !primarySidebar) {
          return;
        }

        event.preventDefault();
        event.stopImmediatePropagation();
        primarySidebar.classList.toggle("pst-sidebar-hidden");
        toggle.blur();
      },
      true,
    );
  });
});
