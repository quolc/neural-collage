canvas_size = 256;

n_zs = 24;
zs = [];
cs = [];

z_source = null;
z_reference = null;
c_source = null;

p5_source = null;
p5_reference = null;
p5_result = null;

function generateSourceNameSpace() {
    return function(s) {
        s.setup = function() {
            s.pixelDensity(1);
            s.createCanvas(canvas_size, canvas_size);

            s.body = null;

            s.cursor(s.HAND);
        }

        s.draw = function() {
            if (s.body != null) {
                s.image(s.body, 0, 0, s.width, s.height);
            }
        }

        s.updateImage = function(url) {
            s.body = s.loadImage(url);
        }
    }
}

function generateReferenceNameSpace() {
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
            if (s.body == null) return;
            s.mask.noStroke();
            s.mask.fill(255, 0, 0);
            s.mask.ellipse(s.mouseX, s.mouseY, 30, 30);
        }

        s.updateImage = function(url) {
            s.body = s.loadImage(url);
        }

        s.clearMask = function(url) {
            s.mask.remove();
            s.mask = s.createGraphics(canvas_size, canvas_size);
        }
    }
}

function generateResultNameSpace() {
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

function randn_bm(sd) {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return sd * Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function updateImageList(urls) {
    $("#image-selection-body").empty();
    $("#image-selection-source").attr("disabled");
    $("#image-selection-reference").attr("disabled");

    for (var i=0; i<urls.length; i++) {
        $("#image-selection-body").append(
            "<li class=\"image-selection-item\" data-index=\"" + i + "\"><img src=\"" + urls[i] + "\"></li>"
        )
    }
    $(".image-selection-item").click(function() {
        $("#image-selection-source").removeAttr("disabled");
        $("#image-selection-reference").removeAttr("disabled");
        $(".image-selection-item.selected").removeClass("selected");
        $(this).addClass("selected");
    });
}

function generateImages() {
    // set cs
    if ($("#class-select").val() == "-1") { // random
        for (var i=0; i<n_zs; i++) {
            cs[i] = Math.floor(Math.random() * dataset_n_classes);
        }
    } else { // selected class
        for (var i=0; i<n_zs; i++) {
            cs[i] = Number($("#class-select").val()) - dataset_offset;
        }
    }

    // set zs
    for (var i=0; i<n_zs; i++) {
        zs[i] = [];
        for (var j=0; j<128; j++) {
            zs[i][j] = randn_bm(1.0);
        }
    }
    
    // send request
    $.ajax({
        type: "POST",
        url: '/generate',
        data: JSON.stringify({"cs" : cs, "zs" : zs}),
        dataType: "json",
        contentType: "application/json",
    }).done(function(data, textStatus, jqXHR) {
        let urls = data['result'];
        
        updateImageList(urls);
    });
}

function enableUI() {
    if (z_source != null && z_reference != null) {
        $('#blending-submit').removeAttr('disabled');
    }
}

function submitBlending() {
    if (z_source == null || z_reference == null || c_source == null) return;

    var canvas = $('#p5-reference canvas')[1];
    var data = canvas.toDataURL('image/png').replace(/data:image\/png;base64,/, '');

    $.ajax({
        type: "POST",
        url: '/blend',
        data: JSON.stringify({"c" : c_source,
                              "z_src" : z_source,
                              "z_ref" : z_reference,
                              "mask" : data,
                              "lambda" : [$("#ex1").val(), $("#ex2").val(), $("#ex3").val(), $("#ex4").val()]}),
        dataType: "json",
        contentType: "application/json",
    }).done(function(data, textStatus, jqXHR) {
        let urls = data['result'];

        $('#exResult').slider({'max' : urls.length-1, "setValue" : urls.length-1});
        $('#exResult').slider('enable');
        p5_result.updateImages(urls);
    });

}

$(function() {
    // set base-class list
    for (var i = dataset_offset; i < dataset_offset + dataset_n_classes; i++) {
        $("#class-select").append(
            "<option value='" + i + "'>" + dataset_keys[i] + "</option>"
        );
    }

    // generation button
    $("#class-submit").click(function() {
        generateImages();
    });

    // use buttons
    $("#image-selection-source").click(function() {
        var index = $(".image-selection-item.selected").data("index");
        z_source = zs[index];
        c_source = cs[index];
        p5_source.updateImage($(".image-selection-item.selected").find("img").attr("src"));

        enableUI();
    });

    $("#image-selection-reference").click(function() {
        var index = $(".image-selection-item.selected").data("index");
        z_reference = zs[index];
        p5_reference.updateImage($(".image-selection-item.selected").find("img").attr("src"));

        enableUI();
    });

    $('#ex1').slider();
    $('#ex2').slider();
    $('#ex3').slider();
    $('#ex4').slider();
    $('#ex5').slider();

    $("#sketch-clear").click(function() {
       p5_reference.clearMask();
    });

    $("#blending-submit").click(function() {
       submitBlending();
    });

    $('#exResult').slider({
        formatter: function(value) {
            return 'interpolation: ' + (value / (16-1)).toFixed(2);
        }
    });
    $('#exResult').slider('disable');
    $("#exResult").change(function() {
        p5_result.changeCurrentImage(parseInt($("#exResult").val()));
    });

    p5_source = new p5(generateSourceNameSpace(), "p5-source");
    p5_reference = new p5(generateReferenceNameSpace(), "p5-reference")
    p5_result = new p5(generateResultNameSpace(), "p5-result");
});
