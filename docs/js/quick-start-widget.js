var opts = {
  cuda: 'cuda11.3',
  os: 'linux',
  pm: 'pip',
  rafbuild: 'preview',
};

var os = $(".os > .option");
var package = $(".package > .option");
var cuda = $(".cuda > .option");
var rafbuild = $(".rafbuild > .option");

os.on("click", function() {
  selectedOption(os, this, "os");
});
package.on("click", function() {
  selectedOption(package, this, "pm");
});
cuda.on("click", function() {
  selectedOption(cuda, this, "cuda");
});
rafbuild.on("click", function() {
  selectedOption(rafbuild, this, "rafbuild")
});

// Pre-select OS
$(function() {
  var userOsOption = document.getElementById(opts.os);
  if (userOsOption) {
    $(userOsOption).trigger("click")
  }
});

function selectedOption(option, selection, category) {
  $(option).removeClass("selected");
  $(selection).addClass("selected");
  opts[category] = selection.id;
  if (category === "pm") {
  }

  commandMessage(buildMatcher());
  if (category === "os") {
    display(opts.os, 'installation', 'os');
  }
}

function display(selection, id, category) {
  var container = document.getElementById(id);
  // Check if there's a container to display the selection
  if (container === null) {
    return;
  }
  var elements = container.getElementsByClassName(category);
  for (var i = 0; i < elements.length; i++) {
    if (elements[i].classList.contains(selection)) {
      $(elements[i]).addClass("selected");
    } else {
      $(elements[i]).removeClass("selected");
    }
  }
}

function buildMatcher() {
  return (
    opts.rafbuild.toLowerCase() +
    "," +
    opts.pm.toLowerCase() +
    "," +
    opts.os.toLowerCase() +
    "," +
    opts.cuda.toLowerCase()
  );
}

function setupMapping() {
  var object = {}
  for (var platform of ["linux", "mac", "win"]) {
    for (var ver of ["preview", "stable"]) {
      for (var cuda of ["none", "11.3"]) {
        const conda_key = ver + ",conda," + platform + ",cuda" + cuda;
        const pip_key = ver + ",pip," + platform + ",cuda" + cuda;

        var raf_name = "raf";
        var tvm_name = "tvm";
        if (ver == "preview") {
          raf_name = raf_name + "-nightly";
        }
        if (cuda != "none") {
          // cuda specific version
          cuda_ver_str = "-cu" + cuda.split(".").join("")
          raf_name = raf_name + cuda_ver_str;
          tvm_name = tvm_name + cuda_ver_str;
        } else {
          raf_name = raf_name + "-cpu";
          tvm_name = tvm_name + "-cpu";
        }
        const conda_enabled = false;
        const pip_enabled = (platform == "linux" && ver == "preview");

        if (pip_enabled) {
          object[pip_key] = "pip install " + raf_name + " " + tvm_name + " -f https://awslabs.github.io/raf/wheels.html";
        }
        if (conda_enabled) {
          object[conda_key] = "conda install " + raf_name + " -c raf";
        }
      }
    }
  }
  return object;
}

var commandMap = setupMapping();

function commandMessage(key) {
  if (!commandMap.hasOwnProperty(key)) {
    $("#command").html(
      "<pre> # Follow instructions at: https://github.com/awslabs/raf/tree/main/docs/wiki/1_getting_start </pre>"
    );
  } else {
    $("#command").html("<pre>" + commandMap[key] + "</pre>");
  }
}
