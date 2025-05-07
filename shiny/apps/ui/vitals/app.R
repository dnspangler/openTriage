library(shiny)
library(jsonlite)
library(lubridate)
library(httr)
library(epiR)

# Set this if you want to use a different framework
fw_name = "amb_refer"

# Set default region
default_region = "region_Uppsala"

# Set this if you're not running this and the openTriage back-end on the same server
server_url = "http://opentriage:5000"


model_props = fromJSON(paste0("../../frameworks/",fw_name,"/models/model_props.json"))
pretty_names = unlist(fromJSON(paste0("../../frameworks/",fw_name,"/models/pretty_names.json")))


feats = data.frame(var = names(model_props$feat_props$gain),
                   gain = unlist(model_props$feat_props$gain),
                   stringsAsFactors = F)

feats = feats[rev(order(feats$gain)),]


# Handle categorical variables
complaint_names <- names(pretty_names)[grepl("complaint_group_",names(pretty_names))]
names(complaint_names) <- pretty_names[match(complaint_names,names(pretty_names))]

region_names <- names(pretty_names)[grepl("region_",names(pretty_names))]
names(region_names) <- pretty_names[match(region_names,names(pretty_names))]


# Define region specific variables
region_variables = list('avpu' = list('region_Uppsala'),
                        'gcs' = list('region_SU','region_SKAS','region_NU','region_SS','region_KUN','region_Uppsala','region_rebro','region_vrmland','region_vstmanland'),
                        'rls' = list('region_Halland','region_rebro','region_vrmland','region_vstmanland'))




ui <- fluidPage(
    
    titlePanel("openTriage - Ambulance referral risk"),
    
    sidebarLayout(
        #actionButton("predict","Predict"),
        sidebarPanel(
            selectInput("region",
                        "Region",
                        choices = region_names,
                  selected = default_region),
            sliderInput("age",
                        "Patient Age",
                        min = 18,
                        max = 100,
                        value = model_props$feat_props$median$age),
            radioButtons("female",
                         "Patient Gender",
                         choices = list("Male"=0,"Female"=1)),

            selectInput("complaint_group",
                        "Complaint",
                        choices = complaint_names,
                        selected = "complaint_group_vrigt"),

            radioButtons("disp_prio1",
                         "Dispatch Priority",
                         choices = list("1"=1,"2-4"=0),
                         selected = model_props$feat_props$median$disp_prio1),

            conditionalPanel(
                condition = "input.region == 'region_Uppsala'",
                radioButtons("avpu",
                         "Level of Consciousness (AVPU)",
                         choices = list("Alert"=1,"Verbal"=2,"Pain"=3,"Unconscious"=4),
                     selected = 1)
            ),
            
            conditionalPanel(
        condition = "['region_SU','region_SKAS','region_NU','region_SS','region_KUN','region_Uppsala','region_rebro','region_vrmland','region_vstmanland'].includes(input.region)",
                sliderInput("gcs",
                        "Level of Consciousness (GCS)",
                        min = 3,
                        max = 15,
                    value = model_props$feat_props$median$gcs)
            ),
            conditionalPanel(
        condition = "['region_Halland','region_rebro','region_vrmland','region_vstmanland'].includes(input.region)",
                sliderInput("rls",
                        "Level of Consciousness (RLS)",
                        min = 1,
                        max = 8,
                    value = model_props$feat_props$median$rls)
            ),
            
            sliderInput("breaths",
                        "Breathing rate",
                        min = 0,
                        max = 50,
                        value = model_props$feat_props$median$breaths),
            sliderInput("spo2",
                        "Oxygen saturation (spo2)",
                        min = 50,
                        max = 100,
                        value = model_props$feat_props$median$spo2),
            sliderInput("sbp",
                        "Systolic blood pressure",
                        min = 0,
                        max = 400,
                        value = model_props$feat_props$median$sbp),
            sliderInput("pulse",
                        "Pulse rate",
                        min = 0,
                        max = 200,
                        value = model_props$feat_props$median$pulse),
            sliderInput("temp",
                        "Temperature",
                        min = 30.5,
                        max = 42.5,
                        value = model_props$feat_props$median$temp),
            textInput("disp_time",
                      "Call time",
                      value = now())
            
        ),
        
        mainPanel(
            tabsetPanel(type = "tabs",
                        tabPanel("Prediction",
                                 htmlOutput("ui")),
                        tabPanel("About",
                                 tagList(
                                     p(),
                                     "This app demonstrates the behavior of a risk assessment instrument reflecting a
                                   patient's risk for deterioration at the time of initial evaluation by an ambulance crew on scene. The instrument 
                                     is based on methods described in our research article", 
                                     a("A validation of machine learning-based risk scores in the prehospital setting", href="https://doi.org/10.1371/journal.pone.0226518"),
                                     ". This user inferface contains no code to estimate risk scores, but rather calls to", 
                                     a("openTriage", href="https://github.com/dnspangler/openTriage"),
                                     ", an open-source API for estimating risk scores for use in clinical decision support systems.
                                     Please note that these scores have not been validated outside of Region Uppsala in Sweden, and 
                                     that this tool is provided for demonstation purposes only. The software is provided 'as is' under the terms of the", 
                                     a("GPLv3 license", href="https://www.gnu.org/licenses/gpl-3.0.en.html"),"
                                      without assumption of liability or warranty of any kind. Put plainly: If you use this on actual patients outside of the 
                                      context in which the models have been validated, you could kill people.",
                                     p(),
                                     "A patient with median values for each predictor included in the models
                                   is described by default. Modify the model parameters in the sidebar and see how the risk assessment
                                   instrument reacts. If no choice is made for the multiple choice values, a missing/other value is assumed.
                                   Multiple choice options are sorted in order of descending average variable importance across all models.",
                                     p(),
                                     "The raw score is displayed for the patient at the top of the screen, and the relative position 
                                     of the score with respect to all scores in a test dataset is displayed. The Likelihoods of each component 
                                     outcome in the score and their percentile ranks are displayed beneath the graph. Finally, average SHAP values 
                                     for each variable across the component outcomed are displayed to explain how the model arrived at the final 
                                     score.",
                                     p(),
                                     "The API this front-end system employs may be accessed via a POST request to https://opentriage.net/predict/amb_refer. 
                                     The API expects a JSON file with a specific format. You can download a test payload based on the currently 
                                     selected predictors here:",
                                     p(),
                                     downloadButton("download",
                                                    "Download test data"),
                                     p(),
                                     "Diagnostics based on",length(model_props$scores)," validation samples at score:",
                                     p(),
                                     sliderInput("diag_score",
                                                 NULL,
                                                 min = min(round(model_props$scores,2)),
                                                 max = max(round(model_props$scores,2)),
                                                 value = 0),
                                     tableOutput("diag"),
                                     p(),
                                     fluidRow(style="text-align: center;padding:60px;",
                                              a(img(src="as.png", width = "200px"), 
                                                href="https://pubcare.uu.se/research/hsr/Ongoing+projects/Political%2C+administrative+and+medical+decision-making/emdai/"),
                                              p(),
                                              a(img(src="uu.png", width = "200px"), 
                                                href="http://ucpr.se/projects/emdai/"),
                                              p(),
                                              a(img(src="vinnova.png", width = "200px"), 
                                                href="https://www.vinnova.se/en/p/emdai-a-machine-learning-based-decision-support-tool-for-emergency-medical-dispatch/")
                                     ))))
        )
    )
)

server <- function(input, output, session) {
    
    # 
    updateTextInput(session, "disp_time", value = format(now(tzone=Sys.timezone()),'%Y-%m-%d %H:%M:%S'))
    
    # Observer to update score and ui page upon changing parameters
    
    observe({
        
        payload <- get_payload(input)
        
        r <- POST(paste0(server_url,"/predict/",fw_name,"/"), 
            content_type_json(), 
            body = payload)

        r_content <- content(r,"parsed")
        
        if(class(r_content) == "list"){
            
            updateSliderInput(session,"diag_score",value = round(r_content$gui$score,2))
            updateSliderInput(session,"diag_percentile",value = round(r_content$gui$score))
            output$ui <- renderUI({
                HTML(as.character(r_content$gui$html))
            })
            
        }else{
            output$ui <- renderUI({
                HTML(as.character(r_content))
            })
            
        }
        
        
    })
    
    get_payload <- function(input) {

        out = list("gui" = list(
            "region"=gsub("region_","",input$region),
            "disp_time"=input$disp_time,
            "age"=as.numeric(input$age),
            "female"=as.numeric(input$female),
            "complaint_group"=gsub("complaint_group_","",input$complaint_group),
            "disp_prio1"=as.numeric(input$disp_prio1),
            "breaths"= input$breaths,
            "spo2"= input$spo2,
            "sbp"= input$sbp,
            "pulse"=input$pulse,
            "temp"=input$temp
        ))

        if(input$region %in% region_variables[["avpu"]]){
            out$gui$avpu <- as.numeric(input$avpu)
        }
        if(input$region %in% region_variables[["gcs"]]){
            out$gui$gcs <- input$gcs
        }
        if(input$region %in% region_variables[["rls"]]){
            out$gui$rls <- input$rls
        }

        return(toJSON(out,pretty = T))
    }
    
    output$download <- downloadHandler(
        
        filename = function() {
            paste(fw_name,'-test-', Sys.Date(), '.json', sep='')
        },
        
        content = function(con) {
            write(get_payload(input), con)
        }
    )
    
    output$diag <- renderTable({
        
        s <- input$diag_score
        
        tt <- lapply(model_props$confusion_matrices,function(x){
            x[[which.min(abs(as.numeric(names(x)) - s))]]
        } )
        
        tt_epi <- lapply(tt,function(x){
            # Need to remap position of TN and TP between python and epiR
            out = x
            out[1,1] = x[2,2]
            out[2,2] = x[1,1]
            return(out)
        })
        
        diags <- lapply(tt_epi,epi.tests)
        
        out <- sapply(diags,function(x){
            sapply(x$rval,function(y){
                round(y$est,2)
            })
        })
        out <- as.data.frame(out,stringsAsFactors = F)
        names(out) <- pretty_names[match(names(out),names(pretty_names))]
        
        return(out)
        
    },rownames = T)
}

# Run the application 
shinyApp(ui = ui, server = server)
