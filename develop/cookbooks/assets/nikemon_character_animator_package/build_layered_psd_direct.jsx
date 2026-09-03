#target photoshop
app.displayDialogs = DialogModes.NO;
var folder = new Folder("C:/work/code/ER-FlowScan/rf-detr/docs/cookbooks/assets/nikemon_character_animator_package");
var parts = [
  {name: "Body", file: "body.png", x: 231, y: 403},
  {name: "+Left Leg", file: "left_leg.png", x: 237, y: 675},
  {name: "+Right Leg", file: "right_leg.png", x: 514, y: 676},
  {name: "+Right Arm", file: "right_arm.png", x: 617, y: 508},
  {name: "+Left Arm", file: "left_arm.png", x: 229, y: 508},
  {name: "+Left Foot", file: "left_foot.png", x: 216, y: 830},
  {name: "+Right Foot", file: "right_foot.png", x: 528, y: 830},
  {name: "+Left Hand", file: "left_hand.png", x: 233, y: 607},
  {name: "+Right Hand", file: "right_hand.png", x: 591, y: 607},
  {name: "+Head", file: "head.png", x: 119, y: 8},
  {name: "+Bell", file: "bell.png", x: 426, y: 526}
];
var doc = app.documents.add(1024, 1024, 72, "nikemon_character_animator_layers", NewDocumentMode.RGB, DocumentFill.TRANSPARENT);
function placeLayer(part) {
    var f = new File(folder.fsName + "/" + part.file);
    app.open(f);
    var partDoc = app.activeDocument;
    partDoc.activeLayer.name = part.name;
    partDoc.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
    partDoc.close(SaveOptions.DONOTSAVECHANGES);
    app.activeDocument = doc;
    var layer = doc.activeLayer;
    layer.name = part.name;
    var bounds = layer.bounds;
    var left = bounds[0].as("px");
    var top = bounds[1].as("px");
    layer.translate(part.x - left, part.y - top);
}
for (var i = 0; i < parts.length; i++) {
    placeLayer(parts[i]);
}
var psdFile = new File(folder.fsName + "/nikemon_character_animator_layers.psd");
var options = new PhotoshopSaveOptions();
options.layers = true;
doc.saveAs(psdFile, options, true, Extension.LOWERCASE);
doc.close(SaveOptions.DONOTSAVECHANGES);
