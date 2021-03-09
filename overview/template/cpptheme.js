
cpptheme = new Theme(
    (str => str.substring(0, str.lastIndexOf("/")))(document.currentScript.src),
    '$BASEURL/style.css',
    {
	thumb: function () {
	    return '.thumb[\n.thumbtxt[\n' + this +'\n]\n]';
	},
	footer: function() {
	    return epcc.footer() + "<p> CC-BY </p>";
	}
    });
