import { app } from "../../../scripts/app.js";

/**
 * After locale overrides wipe backend display_name/description for Comfy Core,
 * re-apply the warning for nodes that use modern widgets (DynamicCombo, etc.).
 * Display-only; does not change node ids or execution.
 */
const SUFFIX = " (new frontend needed)";
const MARKERS = ["COMFY_DYNAMICCOMBO_V3", "IMAGECOMPARE", "AUDIO_RECORD"];

function needsNewFrontend(nodeData) {
	if (!nodeData?.input) return false;
	if (nodeData.api_node) return false;
	const blob = JSON.stringify(nodeData.input);
	return MARKERS.some((m) => blob.includes(m));
}

function withSuffix(text, fallback) {
	const base = (text && String(text).trim()) || fallback || "";
	if (!base) return SUFFIX.trim();
	if (base.includes("new frontend needed")) return base;
	return base + SUFFIX;
}

function withPrefix(text) {
	const base = (text && String(text).trim()) || "";
	if (!base) return "(new frontend needed)";
	if (base.includes("new frontend needed")) return base;
	return `(new frontend needed) ${base}`;
}

app.registerExtension({
	name: "frontend_needed_labels",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (!needsNewFrontend(nodeData)) return;
		nodeData.display_name = withSuffix(nodeData.display_name, nodeData.name);
		nodeData.description = withPrefix(nodeData.description);
	},
});
