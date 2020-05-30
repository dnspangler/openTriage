
library(shiny)
library(jsonlite)
library(lubridate)
library(httr)
library(epiR)

# Set this if you're using a different framework
fw_name = "uppsala_vitals"

# Set this if you're not running this and the openTriage back-end on the same server
server_url = "http://opentriage:5000"
cat(file=stdout(),"Hello!")
model_props = fromJSON(paste0("../../frameworks/",fw_name,"/models/model_props.json"))
pretty_names = unlist(fromJSON(paste0("../../frameworks/",fw_name,"/models/pretty_names.json")))


feats = data.frame(var = names(model_props$feat_props$gain),
                   gain = unlist(model_props$feat_props$gain),
                   stringsAsFactors = F)

feats = merge(feats,
              data.frame(var = names(pretty_names),
                         name = pretty_names),
              stringsAsFactors = F)

feats = feats[rev(order(feats$gain)),]

cat_names = gsub("disp_cats_","",feats$var)[grepl("disp_cats_",feats$var)]
names(cat_names) = feats$name[grepl("disp_cats_",feats$var)]

ui_page = reactiveVal()

ui <- fluidPage(
    
    titlePanel("openTriage - Uppsala Vitals demo"),
    
    sidebarLayout(
        #actionButton("predict","Predict"),
        sidebarPanel(
            selectInput("region",
                        "Region",
                        choices = c("Uppsala")),
            sliderInput("disp_age",
                        "Patient Age",
                        min = 0,
                        max = 100,
                        value = model_props$feat_props$median$disp_age),
            radioButtons("disp_gender",
                         "Patient Gender",
                         choices = list("Male"=0,"Female"=1)),
            selectInput("disp_cats",
                        "Dispatch Categories",
                        choices = cat_names,
                        multiple = T),
            radioButtons("disp_prio",
                         "Dispatch Priority",
                         choices = list("1A"=1,"1B"=2,"2A"=3,"2B"=4,"Referral"=7),
                         selected = model_props$feat_props$median$disp_prio),
            radioButtons("eval_avpu",
                         "Level of Consciousness (AVPU)",
                         choices = list("Alert"="A","Verbal"="V","Conscious"="P","Unconscious"="U"),
                         selected = "A"),
            sliderInput("eval_breaths",
                        "Breathing rate",
                        min = 0,
                        max = 50,
                        value = model_props$feat_props$median$eval_breaths),
            sliderInput("eval_spo2",
                        "Oxygen saturation (spo2)",
                        min = 50,
                        max = 100,
                        value = model_props$feat_props$median$eval_spo2),
            sliderInput("eval_sbp",
                        "Systolic blood pressure",
                        min = 0,
                        max = 400,
                        value = model_props$feat_props$median$eval_sbp),
            sliderInput("eval_pulse",
                        "Pulse rate",
                        min = 0,
                        max = 200,
                        value = model_props$feat_props$median$eval_pulse),
            sliderInput("eval_temp",
                        "Temperature",
                        min = 30.5,
                        max = 42.5,
                        value = model_props$feat_props$median$eval_temp),
            textInput("disp_created",
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
                                     "This app demonstrates the behaviour of a risk assessment instrument reflecting a
                                   patient's risk for deterioration at the time of initial evaluation by an ambulans crew on scene. The insrument 
                                     is based on methods described in our research article", 
                                     a("A validation of machine learning-based risk scores in the prehospital setting", href="https://doi.org/10.1371/journal.pone.0226518"),
                                     ". This user inferface contains no code to estimate risk scores, but rather calls to", 
                                     a("openTriage", href="https://github.com/dnspangler/openTriage"),
                                     ", an open-source API for estimating risk scores for use in clinical decision support systems.
                                     Please note that these scores have not been validated outside of Region Uppsala in Sweden, and 
                                     that this tool is provided for demonstation purposes only. The software is provided 'as is' under the terms of the", 
                                     a("GPLv3 license", href="https://www.gnu.org/licenses/gpl-3.0.en.html"),"
                                      without assuption of liability or warranty of any kind. Put plainly: If you use this on actual patients outside of the 
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
                                     "The API this front-end system employs may be accessed via a POST request to https://opentriage.net/predict/uppsala_vitals. 
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
    updateTextInput(session, "disp_created", value = now())

    sessionID = paste0(do.call(paste0, replicate(5, sample(c(LETTERS,0:9), 8, TRUE), FALSE)),collapse="")
    
    # Observer to update score and ui page upon changing parameters
    observe({
        
        requestID = paste0(do.call(paste0, replicate(5, sample(c(LETTERS,0:9), 8, TRUE), FALSE)),collapse="")
        
        payload <- get_payload(input)
        
        r <- POST(paste0(server_url,"/predict/",fw_name,"/"), 
            content_type_json(), 
            body = payload,
            add_headers(ids = requestID))

        # Appears that POST requests from other concurrent users are updating eachothers instances, but I can't reproduce the issue locally to track down why... 
        # Adding a session ID to the request header is a kind of hacky fix until I can figure out a proper solution.
        if(headers(r)$ids == requestID){
            r_content <- content(r,"parsed")
        
        if(class(r_content) == "list"){
            
            updateSliderInput(session,"diag_score",value = round(r_content[[sessionID]]$score,2))
            
            ui_page(HTML(as.character(r_content[[sessionID]]$html)))
            
        }else{
            
            ui_page(HTML(as.character(r_content)))
            
        }
        }
        
        
    })
    
    
    get_payload <- function(input) {
        
        data = list(
            "region"=input$region,
            "disp_created"=input$disp_created,
            "disp_age"=input$disp_age,
            "disp_gender"=as.numeric(input$disp_gender),
            "disp_cats"=paste(input$disp_cats,collapse="|"),
            "disp_prio"=as.numeric(input$disp_prio),
            "eval_breaths"= input$eval_breaths,
            "eval_spo2"= input$eval_spo2,
            "eval_sbp"= input$eval_sbp,
            "eval_pulse"=input$eval_pulse,
            "eval_avpu"=input$eval_avpu,
            "eval_temp"=input$eval_temp
        )

        out <- list()

        out[[sessionID]] <- data

        out <- toJSON(out,pretty = T)
        
        return(out)
    }
    
    output$download <- downloadHandler(
        
        filename = function() {
            paste('test', Sys.Date(), '.json', sep='')
        },
        
        content = function(con) {
            write(get_payload(input), con)
        }
    )
    
    output$ui <- renderUI({
        ui_page()
    })
    
    output$diag <- renderTable({
        
        p <- input$diag_score
        
        
        tt <- lapply(model_props$confusion_matrices,function(x){
            x[[which.min(abs(as.numeric(names(x)) - p))]]
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
