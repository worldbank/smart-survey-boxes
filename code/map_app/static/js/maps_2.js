var map = L.map( 'map', {
  center: [38.574875,68.812003],
  minZoom: 2,
  zoom: 12
})

L.tileLayer( 'http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  subdomains: ['a', 'b', 'c']
}).addTo( map )


$.getJSON('/boxes', function(json) {
    var data = json;
  	for (var i = 0; i < data.length; i++) {
    	var item = data[i];
      var xy  = [parseFloat(item.geometry.coordinates[0]),parseFloat(item.geometry.coordinates[1])];
   	 	var circle = L.circle(xy, {
          color: 'green',
          fillColor: 'green',
          fillOpacity: 0.5,
          radius: 300
      }).addTo(map);
	}
});


//grab clicked lat_lon and send to python backend for predicting outage
map.on('click', function(e) {
  var clicked_latlon = {'lat': e.latlng.lat, 'lon': e.latlng.lng};
  $.ajax
  ({
    type: "POST",
    contentType: "application/json; charset=utf-8",
    //the url where you want to sent the userName and password to
    url: '/click_loc',
    async: false,
    //json object to sent to the authentication url
    data: JSON.stringify(clicked_latlon),
    
    success: function () {
      alert("Do you want to see power outage stats for this location?"); 
    }
})

});

$.getJSON("/click_loc", function(json) {
  var loc = json;
  alert("Did power go off in the past one day: " + loc); 
    
});





