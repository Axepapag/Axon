export function analyzeNativeText(text, alphabet) {
  const native = typeof alphabet === "string" ? alphabet : "";
  const invalid = [];
  let offset = 0;
  for (const character of Array.from(String(text ?? ""))) {
    if (!native.includes(character)) invalid.push({offset, character, codePoint: character.codePointAt(0)});
    offset += character.length;
  }
  return {valid: Boolean(native) && invalid.length === 0, invalid};
}

export function displayCharacter(character) {
  if (character === "\n") return "\\n";
  if (character === "\r") return "\\r";
  if (character === "\t") return "\\t";
  if (character === " ") return "space";
  return character;
}

export function defaultsFromSchema(schema) {
  if (!schema || typeof schema !== "object") return {};
  if (Object.prototype.hasOwnProperty.call(schema, "default")) return structuredClone(schema.default);
  if (schema.type === "object" || schema.properties) {
    const value = {};
    for (const [key, child] of Object.entries(schema.properties || {})) {
      const childDefault = defaultsFromSchema(child);
      if (childDefault !== undefined) value[key] = childDefault;
    }
    return value;
  }
  if (schema.type === "array") return Array.isArray(schema.default) ? structuredClone(schema.default) : [];
  return undefined;
}

export function getAtPath(root, path) {
  if (!path) return root;
  return path.split(".").reduce((value, key) => value?.[key], root);
}

export function setAtPath(root, path, value) {
  const parts = path.split(".").filter(Boolean);
  if (!parts.length) return value;
  let cursor = root;
  for (const key of parts.slice(0, -1)) {
    if (!cursor[key] || typeof cursor[key] !== "object" || Array.isArray(cursor[key])) cursor[key] = {};
    cursor = cursor[key];
  }
  cursor[parts.at(-1)] = value;
  return root;
}

export function registrationPayload(architecture) {
  return {
    version: architecture?.version || "draft-1",
    name: architecture?.name || "Untitled experiment",
    nodes: structuredClone(architecture?.nodes || []),
    edges: structuredClone(architecture?.edges || []),
    ports: structuredClone(architecture?.ports || []),
  };
}

export function stableFingerprint(value) {
  const normalize = item => {
    if (Array.isArray(item)) return item.map(normalize);
    if (item && typeof item === "object") {
      return Object.fromEntries(Object.keys(item).sort().map(key => [key, normalize(item[key])]));
    }
    return item;
  };
  return JSON.stringify(normalize(value));
}


function sameJsonValue(left, right) {
  return JSON.stringify(left ?? null) === JSON.stringify(right ?? null);
}

export function portCompatibility(source, destination) {
  const reasons = [];
  if (!source || !destination) return {compatible: false, reasons: ["Both ports are required."]};
  if (source.direction !== "output") reasons.push("Source port is not an output.");
  if (destination.direction !== "input") reasons.push("Destination port is not an input.");

  for (const field of ["meaning", "dtype", "authority", "surface"]) {
    if ((source[field] ?? null) !== (destination[field] ?? null)) {
      reasons.push(`${field} mismatch: ${source[field] ?? "unset"} -> ${destination[field] ?? "unset"}`);
    }
  }
  if (!sameJsonValue(source.shape, destination.shape)) {
    reasons.push(`shape mismatch: ${JSON.stringify(source.shape)} -> ${JSON.stringify(destination.shape)}`);
  }

  if (source.surface === "substrate-exact" || destination.surface === "substrate-exact") {
    if (source.surface !== "substrate-exact" || destination.surface !== "substrate-exact") {
      reasons.push("substrate-exact can connect to free only through an explicit adapter component.");
    } else {
      const sourceCodebook = source.constraints?.codebook ?? source.constraints?.codebook_id ?? null;
      const destinationCodebook = destination.constraints?.codebook ?? destination.constraints?.codebook_id ?? null;
      if (!sourceCodebook || !destinationCodebook) {
        reasons.push("substrate-exact ports must declare a codebook identity.");
      } else if (sourceCodebook !== destinationCodebook) {
        reasons.push(`codebook mismatch: ${sourceCodebook} -> ${destinationCodebook}`);
      }
      const sourceTolerance = source.constraints?.tolerance;
      const destinationTolerance = destination.constraints?.tolerance;
      if (sourceTolerance !== undefined && sourceTolerance !== 0) reasons.push("source substrate tolerance must be 0.0.");
      if (destinationTolerance !== undefined && destinationTolerance !== 0) reasons.push("destination substrate tolerance must be 0.0.");
    }
  }

  return {compatible: reasons.length === 0, reasons};
}
