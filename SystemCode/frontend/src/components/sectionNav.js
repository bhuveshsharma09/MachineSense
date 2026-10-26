// App-8.6 items 18–19: section jumps land below the sticky header and jump bar, and the jump bar follows the scroll
// position (not IntersectionObserver, which lost track after switching Mechanic ↔ Developer).
import { useEffect, useState } from "react";
import { activeSection } from "../lib/dashboard.js";

// the height covered by the sticky bars at the top of the viewport
export function stickyOffset() {
  if (typeof document === "undefined") return 0;
  return [...document.querySelectorAll(".rd-header, .mvw-jump, .dvw-jump")]
    .filter((el) => getComputedStyle(el).position === "sticky")
    .reduce((m, el) => Math.max(m, el.getBoundingClientRect().bottom), 0);
}

export function scrollToSection(id) {
  const el = typeof document !== "undefined" && document.getElementById(id);
  if (!el) return;
  const top = el.getBoundingClientRect().top + window.scrollY - stickyOffset() - 12;
  window.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
}

export function useSectionSpy(ids) {
  const [current, setCurrent] = useState(ids[0]);
  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    let raf = 0;
    const update = () => {
      raf = 0;
      const tops = ids.map((id) => ({ id, el: document.getElementById(id) })).filter((x) => x.el)
        .map(({ id, el }) => ({ id, top: el.getBoundingClientRect().top }));
      const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
      const next = activeSection(tops, stickyOffset(), atBottom);
      if (next) setCurrent(next);
    };
    const on = () => { if (!raf) raf = requestAnimationFrame(update); };
    update();
    window.addEventListener("scroll", on, { passive: true });
    window.addEventListener("resize", on);
    return () => { window.removeEventListener("scroll", on); window.removeEventListener("resize", on); if (raf) cancelAnimationFrame(raf); };
  }, [ids.join()]);
  return [current, setCurrent];
}
