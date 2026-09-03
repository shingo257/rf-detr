#target photoshop
app.displayDialogs = DialogModes.NO;

var scriptFile = new File($.fileName);
var folder = scriptFile.parent;
var manifestFile = new File(folder.fsName + "/manifest.json");
manifestFile.open("r");
var manifest = JSON.parse(manifestFile.read());
manifestFile.close();

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
    layer.translate(part.bbox[0] - left, part.bbox[1] - top);
}

manifest.parts.sort(function(a, b) { return a.z - b.z; });
for (var i = 0; i < manifest.parts.length; i++) {
    placeLayer(manifest.parts[i]);
}

var psdFile = new File(folder.fsName + "/nikemon_character_animator_layers.psd");
var options = new PhotoshopSaveOptions();
options.layers = true;
doc.saveAs(psdFile, options, true, Extension.LOWERCASE);
alert("Created layered PSD: " + psdFile.fsName);
