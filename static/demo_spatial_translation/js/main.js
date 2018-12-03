canvas_size = 256;

max_colors = 12;
colors = ["#FE2712",
          "#66B032",
          "#FEFE33",
          "#FC600A",
          "#0247FE",
          "#B2D732",
          "#4424D6",
          "#FB9902",
          "#8601AF",
          "#FCCC1A",
          "#347C98",
          "#C21460"];

z = null;
base_class = null;
palette = [];

ui_uninitialized = true;
p5_input = null;
p5_output = null;

function generateInputNameSpace() {
    return function(s) {
        s.setup = function() {
            s.pixelDensity(1);
            s.createCanvas(canvas_size, canvas_size);

            s.mask = s.createGraphics(canvas_size, canvas_size);
            s.body = null;

            s.cursor(s.HAND);
        }

        s.draw = function() {
            s.background(255);
            s.noTint();
            if (s.body != null) {
                s.image(s.body, 0, 0, s.width, s.height);
            }
            s.tint(255, 127);
            s.image(s.mask, 0, 0);
        }

        s.mouseDragged = function() {
            if (ui_uninitialized) return;

            var c = $('.palette-item.selected').data('class');
            if (c >= 0) {
                var col = s.color(colors[palette.indexOf(c)]);
                s.mask.noStroke();
                s.mask.fill(col);
                s.mask.ellipse(s.mouseX, s.mouseY, 20, 20);
            } else { // eraser
                var col = s.color(0, 0);
                s.mask.loadPixels();
                for (var x=Math.max(0, Math.floor(s.mouseX) - 10); x<Math.min(canvas_size, Math.floor(s.mouseX) + 10); x++) {
                    for (var y=Math.max(0, Math.floor(s.mouseY) - 10); y<Math.min(canvas_size, Math.floor(s.mouseY) + 10); y++) {
                        if (s.dist(s.mouseX, s.mouseY, x, y) < 10) {
                            s.mask.set(x, y, col);
                        }
                    }
                }
                s.mask.updatePixels();
                // p5.Graphics object should be re-created because of a bug related to updatePixels().
                var new_g = s.createGraphics(canvas_size, canvas_size);
                new_g.image(s.mask, 0, 0);
                s.mask.remove();
                s.mask = new_g;
            }
        }

        s.clear_canvas = function() {
            s.mask.clear();
        }

        s.updateImage = function(url) {
            s.body = s.loadImage(url);
        }
    }
}

function generateOutputNameSpace() {
    return function(s) {
        s.setup = function() {
            s.pixelDensity(1);
            s.createCanvas(canvas_size, canvas_size);

            s.images = [];
            s.currentImage = 0;
            s.frameRate(15);
        }

        s.draw = function() {
            if (s.images.length > s.currentImage) {
                s.background(255);
                s.image(s.images[s.currentImage], 0, 0, s.width, s.height);
            }
        }

        s.updateImages = function(urls) {
            for (var i=urls.length-1; i>=0; i--) {
                var img = s.loadImage(urls[i]);
                s.images[i] = img;
            }
            s.currentImage = urls.length - 1;
        }

        s.changeCurrentImage = function(index) {
            if (index < s.images.length) {
                s.currentImage = index;
            }
        }
    }
}

// box-muller
function randn_bm(sd) {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return sd * Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function generateNewImage() {
    // generate new random z
    z = [];
    N = 128;
    SD = 1.0;
    for (var i=0; i<N; i++) {
        z.push(randn_bm(SD));
    }

    // clear class-map
    p5_input.clear_canvas();

    // request
    updateResult(true);
}

function updateResult(update_both) {
    var canvas = $('#p5-left canvas')[1];
    var data = canvas.toDataURL('image/png').replace(/data:image\/png;base64,/, '');

    var palette_nooffset = [];
    for (var i in palette) {
        palette_nooffset.push(palette[i] - dataset_offset)
    }

    $.ajax({
        type: "POST",
        url: "/post",
        data: JSON.stringify({"c" : base_class - dataset_offset, "palette" : palette_nooffset,
                              "z" : z, "class_map" : data, "colors" : colors}),
        dataType: "json",
        contentType: "application/json",
    }).done(function(data, textStatus, jqXHR) {
        enableUI();
        let urls = data['result'];

        // update left panel
        if (update_both) {
            p5_input.updateImage(urls[0]);
        }
        $('#ex1').slider({'max' : urls.length-1, "setValue" : urls.length-1});
        p5_output.updateImages(urls);
    });
}

function enableUI() {
    ui_uninitialized = false;
    $("#sketch-clear").removeAttr('disabled');
    $("#main-ui-submit").removeAttr('disabled');
    $('#ex1').slider('enable');
}

$(function(){
    // set base-class list
    for (var i=dataset_offset; i<dataset_offset+dataset_n_classes; i++) {
        $("#base-class-select").append(
            "<option value='" + i + "'>" + dataset_keys[i] + "</option>"
        );
    }

    // generation
    $("#base-class-submit").click(function() {
        base_class = $("#base-class-select").val();
        generateNewImage();
    });

    // main
    $("#main-ui-submit").click(function() {
        updateResult();
    });

    $("#sketch-clear").click(function() {
        p5_input.clear_canvas();
        $('.palette-item-class').remove();
        palette = [];
        $("#palette-eraser").click();
    });

    // setup class-picker
    for (var i=dataset_offset; i<dataset_offset+dataset_n_classes; i++) {
        var path = sprintf("/static/demo_spatial_translation/img/" + dataset_name + "/%03d.jpg", i);
        $("#class-picker").append(
            '<option data-img-src="' + path + '" data-img-alt="' + dataset_keys[i] + '" value="' + (i) + '">' + dataset_keys[i] + '</option>'
        );
    }

    $("#class-picker").imagepicker({
        hide_select: false,
    });
    $('#class-picker').after(
        "<button type=\"submit\" class=\"form-control btn btn-primary col-md-2\" id=\"class-picker-submit\">add to palette</button>"
    );
    $("#class-picker-submit").after(
        "<div class=\"row\" id=\"class-picker-ui\"></div>"
    )
    $("#class-picker").appendTo("#class-picker-ui");
    $("#class-picker-submit").appendTo("#class-picker-ui");

    $("#class-picker-submit").click(function() {
        const selected_class = $("#class-picker").val();
        if (palette.length >= max_colors || palette.indexOf(Number(selected_class)) != -1) {
            return;
        }

        $("#palette-body").append(
            "<li class=\"palette-item palette-item-class\" id=\"palette-" + selected_class + "\"" +
            "data-class=\"" + selected_class +
            "\" style=\"background-color: " + colors[palette.length] + ";\"></li>");
        $("#palette-" + selected_class).append(
            "<img src=\"/static/demo_spatial_translation/img/" + dataset_name + "/" + selected_class + ".jpg\">"
        )

        palette.push(Number(selected_class));

        $("#palette-" + selected_class).click(function() {
            $(".palette-item.selected").removeClass('selected');
            $(this).addClass('selected');
        });
        $("#palette-" + selected_class).click();
    });
    $("#palette-eraser").click(function() {
        $(".palette-item.selected").removeClass('selected');
        $(this).addClass('selected');
    });

    $('#ex1').slider({
        formatter: function(value) {
            return 'interpolation: ' + (value / (16-1)).toFixed(2);
        }
    });
    $('#ex1').slider('disable');
    $("#ex1").change(function() {
        p5_output.changeCurrentImage(parseInt($("#ex1").val()));
    });

    p5_input = new p5(generateInputNameSpace(), "p5-left");
    p5_output = new p5(generateOutputNameSpace(), "p5-right");
})