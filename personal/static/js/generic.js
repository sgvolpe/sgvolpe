



// When the user scrolls down 50px from the top of the document, resize the header's font size
window.onscroll = function() {scrollFunction()};

function scrollFunction() {
  if (document.body.scrollTop > 50 || document.documentElement.scrollTop > 50) {
    document.getElementById("header").style.fontSize = "12px";
    $('#name').html('SGV') ;
    $('#name').css("font-family", "Bungee Shade, cursive;") ;
    $('nav').css("color","#f0f8ff00") ;
    $('#name').parent().css("margin","") ;
    $('#chartdiv').css("opacity","0.5") ;
    // // logo
  } else {
    document.getElementById("header").style.fontSize = "30px";
    $('#name').html('Santiago Gonzalez Volpe') ;
    $('#name').parent().css("margin","auto") ;
    $('#chartdiv').css("opacity","1") ;


  }


} ;



am4core.ready(function() {

  // Themes begin
  am4core.useTheme(am4themes_animated);
  // Themes end


  var chart = am4core.create("chartdiv", am4plugins_wordCloud.WordCloud);
  var series = chart.series.push(new am4plugins_wordCloud.WordCloudSeries());

  series.accuracy = 4;
  series.step = 15;
  series.rotationThreshold = 0.7;
  series.maxCount = 200;
  series.minWordLength = 2;
  //series.labels.template.tooltipText = "{word}: {value}";
  series.fontFamily = "Courier New";
  series.maxFontSize = am4core.percent(30);

  series.text = "Provided training team both product  analytics.Completi 12+ customer engagements, including top OTAs EMEA.Bundled vs unbundled Analysis.Stablished process  Shopping Benchmarking, increasing volume  data as analysis capabilities available Automation Ancillaries AnalysisAwarded Team  yearSupported Americas OTAs portfolio. Development  tools OCEAN, including Automati support processes, lead winning 2016 GO Team Excellence Award consisted  cash price  USD4000 team were able create set  tools used across Sabre enterprise Took part migrating major OTAs EMEA. Winning 2nd Half 2017 Champion’s League Team Excellence Award collaborated ensure this account started producing bookings weeks  kick-off! You demonstrated impeccable teamwork  persistence  how Excellence Executive deliver results Support  Abacus during migration Shopping Specialist 2nd level.Creation documentation, extensive  intensive about Shopping products  Creating  desktop applications Helpdesk. Pyth(experienced)JAVA Php (basic) Javascript MySQL, Teradata Git. Microst AI School: PythData Science Machine  Crash Course UDACITY  Data Analysis Data Camp ductiData Visualization Python Udemy Python Data Structures, Algorithms,  Interviews. Interactive PythDashboards Plotly  Dash  Pyth Django Full Stack Web Developer Bootcamp Python Data Science  Machine  Bootcamp Deep  Prerequisites: Numpy Stack Python Web-Programming Writing Customer Service Emails Python Data Structures: Stacks Queues Deques Communicating Confidence SQL Essential Training Leadership Competency Highlights Business Strategy  Analysis Consulting Foundations Client Management Relationships Customer Advocacy Customer Service Working Customer Contact Center Decision-Making Strategies New Manager Foundations Word Essential Training Office-365 Khan-Academy  HTML CSS HTML JavaScript SQL Advanced JavaScript";

});






console.log('js loaded');
