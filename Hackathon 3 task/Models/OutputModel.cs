using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace Hackathon_3_task.Models
{
    public class OutputModel
    {
        public OutputModel()
        {
            intents = new List<Intent>();
        }
        public string id { get; set; }
        public List<Intent> intents { get; set; }
    }

    public class OutputModelWithText
    {
        public OutputModelWithText()
        {
            intents = new List<Intent>();
        }
        public string id { get; set; }
        public string text { get; set; }
        public List<Intent> intents { get; set; }
    }

    public class Intent
    {
        public string text { get; set; }
        public int startsAt { get; set; }
    }
}