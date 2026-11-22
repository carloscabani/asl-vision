// Frases cuya traduccion prefieres fija (coincidencia exacta del texto completo,
// en minusculas). Un traductor automatico dice "hello" para "hola"; aqui
// puedes forzar el registro informal. Formato: "origen>destino": { texto: traduccion }
export const OVERRIDES = {
  "es>en": { hola: "hi" },
  "en>es": { hi: "hola" },
};
