// Dialogs and notifications. Dialogs use the native <dialog> element, so focus
// is trapped and Escape closes them; every message carries an icon and text.

import { h, icon } from "./dom.js";

const STATUS_ICON = { ok: "check", error: "cross", warn: "warn", info: "dot" };

export function toast(text, { kind = "info", action = null, timeout = 5000 } = {}) {
  let host = document.getElementById("toasts");
  if (!host) {
    host = h("div", { id: "toasts", class: "toasts", role: "status", "aria-live": "polite" });
    document.body.append(host);
  }
  const item = h("div", { class: `toast ${kind}` }, [icon(STATUS_ICON[kind] || "dot"), h("span", { text })]);
  if (action) {
    item.append(h("button", { type: "button", class: "linkish", text: action.label, onclick: () => {
      item.remove();
      action.run();
    } }));
  }
  item.append(h("button", { type: "button", class: "icon-btn", "aria-label": "Dismiss", onclick: () => item.remove() }, [icon("cross")]));
  host.append(item);
  if (timeout) setTimeout(() => item.remove(), kind === "error" ? timeout * 2 : timeout);
  return item;
}

export function dialog({ title, body, actions = [{ label: "Close", value: null, primary: true }], wide = false }) {
  return new Promise(resolve => {
    const box = h("dialog", { class: "dialog" + (wide ? " wide" : ""), "aria-label": title });
    const form = h("form", { method: "dialog" });
    const footer = h("div", { class: "dialog-actions" });
    for (const [index, action] of actions.entries()) {
      const value = String(index);
      const button = h("button", { type: action.submit === false ? "button" : "submit", class: "btn" + (action.primary ? " primary" : ""), value, text: action.label });
      if (action.submit === false) button.addEventListener("click", () => box.close(value));
      footer.append(button);
    }
    form.append(h("h2", { text: title }), h("div", { class: "dialog-body" }, [].concat(body)), footer);
    box.append(form);
    document.body.append(box);
    box.addEventListener("close", () => {
      const index = Number(box.returnValue);
      const action = Number.isInteger(index) && box.returnValue !== "" ? actions[index] : null;
      box.remove();
      resolve(action ? (typeof action.value === "function" ? action.value() : action.value) : null);
    });
    box.showModal();
    const first = box.querySelector("input, textarea, select");
    if (first) first.focus();
  });
}

export async function promptDialog({ title, label, value = "", placeholder = "", confirm = "OK", help = "" }) {
  const input = h("input", { type: "text", class: "input", value, placeholder, "aria-label": label, spellcheck: "false" });
  const body = [h("label", { class: "dialog-label", text: label }), input];
  if (help) body.push(h("p", { class: "help", text: help }));
  const result = await dialog({
    title,
    body,
    actions: [
      { label: "Cancel", value: null, submit: false },
      { label: confirm, value: () => input.value, primary: true },
    ],
  });
  return result;
}

export async function confirmDialog({ title, text, confirm = "Continue", cancel = "Cancel" }) {
  const result = await dialog({ title, body: [h("p", { text })], actions: [{ label: cancel, value: false }, { label: confirm, value: true, primary: true }] });
  return result === true;
}
