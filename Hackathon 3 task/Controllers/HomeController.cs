﻿using Hackathon_3_task.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using Newtonsoft.Json;

namespace Hackathon_3_task.Controllers
{
    public class HomeController : Controller
    {
        public ActionResult Index()
        {
            return RedirectToAction("Intent");
        }

        public ActionResult Intent()
        {
            return View();
        }


        //Only text
        [HttpPost]
        public ActionResult ResultWithText(string Text)
        {
            string result = PythonExecute(Text);
            List <Intent> IntentsResult = new List<Intent>();
            var intents = result.Split( new string[] { "###" }, StringSplitOptions.None);
            //Окей, тут конверт треба написати в залежності від входу (не єбу який він)
            foreach (var c in intents)
            {
                if (!string.IsNullOrEmpty(c))
                {
                    IntentsResult.Add(new Intent { startsAt = Text.IndexOf(c), text = c });
                }
            }

            if (IntentsResult.Count == 0)
            {
                ViewBag.Zero = true;
            }
            OutputModelWithText outputModelWithText = new OutputModelWithText();
            outputModelWithText.intents = IntentsResult;
            outputModelWithText.text = Text;
            return View(outputModelWithText);
        }

        //Only for SendIntent
        //public JsonResult GetJson(List<Intent> intents, string text)
        //{
        //    OutputModel outputModel = 
        //    var jsondata = db.Books.Where(a => a.Author.Contains(name)).ToList<Book>();
        //    return Json(jsondata, JsonRequestBehavior.AllowGet);
        //}


        [NonAction]
        public string PythonExecute(string text)
        {
            ProcessStartInfo start = new ProcessStartInfo();

            // full path of python exe
            start.FileName = @"G:\hackathon\python.exe";

            string cmd = @"G:\hackathon python\untitled1\script.py";
            string args = text;

            // define the script with arguments (if you need them).
            start.Arguments = string.Format("\"{0}\" \"{1}\"", cmd, args);

            // Do not use OS shell
            start.UseShellExecute = false;

            // You do not need new window 
            start.CreateNoWindow = true;

            // Any output, generated by application will be redirected back
            start.RedirectStandardOutput = true;

            // Any error in standard output will be redirected back (for example exceptions)
            start.RedirectStandardError = true;

            // start the process
            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    // Here are the exceptions from our Python script
                    string stderr = process.StandardError.ReadToEnd();

                    // Here is the result of StdOut(for example: print "test")
                    string result = reader.ReadToEnd();

                    return result;
                }
            }
        }

        [HttpPost]
        public ActionResult Upload()
        {
            bool isSavedSuccessfully = true;
            string fName = "";
            List<string> list = new List<string>();
            try
            {
                foreach (string fileName in Request.Files)
                {
                    HttpPostedFileBase file = Request.Files[fileName];
                    fName = file.FileName;
                    if (file != null && file.ContentLength > 0)
                    {
                        var path = Path.Combine(Server.MapPath("~/Json Files"));
                        string pathString = Path.Combine(path.ToString());
                        string fileName1 = GetGUID(file.FileName);
                        bool isExists = Directory.Exists(pathString);
                        if (!isExists) Directory.CreateDirectory(pathString);
                        var uploadpath = string.Format("{0}\\{1}", pathString, fileName1);
                        file.SaveAs(uploadpath);
                        //string URL = Request.Url.Scheme + "://" + Request.Url.Authority + "/Json Files";
                        list.Add(path + "\\" + fileName1);

                    }
                }
            }
            catch
            {
                isSavedSuccessfully = false;
            }
            if (isSavedSuccessfully)
            {
                //return RedirectToAction("SendCheck", list.First());
                return Json(new
                {
                    Message = fName,
                    MyFiles = list
                });
            }
            else
            {
                return Json(new
                {
                    Message = "Error in saving files"
                });
            }
        }

        [HttpPost]
        public ActionResult SendIntent(string fileName)
        {
            if (string.IsNullOrEmpty(fileName))
                return RedirectToAction("Index");
            string rezult;
            using (StreamReader reader = new StreamReader(fileName))
            {
                rezult = reader.ReadToEnd();
            }
            //тут файл дописати 
            //var inputJSONModel = Newtonsoft.Json.JsonConvert.DeserializeObject<InputJSONModel>(rezult);
            //foreach (var c in inputJSONModel)
            //{

            //}
            //var jsondata = outputModel;
            //call python
            var mydictionary = JsonConvert.DeserializeObject<List<InputJSONModel>>(rezult);
            List<OutputModel> listOutputModel = new List<OutputModel>();
            foreach (var mydict in mydictionary)
            {
                string text = mydict.text.Replace('"', ' ');
                string result = PythonExecute(text);

                List<Intent> IntentsResult = new List<Intent>();
                var intents = result.Split(new string[] { "###" }, StringSplitOptions.None);
                //Окей, тут конверт треба написати в залежності від входу (не єбу який він)
                foreach (var c in intents)
                {
                    if (!string.IsNullOrEmpty(c))
                    {
                        IntentsResult.Add(new Intent { startsAt = text.IndexOf(c), text = c });
                    }
                }

                listOutputModel.Add(new OutputModel { id = mydict.id, intents = IntentsResult });
            }
            ViewBag.MyJSON = Json(listOutputModel, JsonRequestBehavior.AllowGet);
            ViewBag.fileName = fileName;
            List<OutputModelWithText> list = new List<OutputModelWithText>();
            foreach (var c in listOutputModel)
            {
                list.Add(new OutputModelWithText
                {
                    id = c.id,
                    intents = listOutputModel.Where(f => f.id.Equals(c.id)).First().intents,
                    text = mydictionary.Where(f => f.id == c.id).First().text
                });
            }
            return View(list);
        }

        [HttpPost]
        public JsonResult GetJSON(string fileName)
        {
            string rezult;
            using (StreamReader reader = new StreamReader(fileName))
            {
                rezult = reader.ReadToEnd();
            }
            var mydictionary = JsonConvert.DeserializeObject<List<InputJSONModel>>(rezult);
            List<OutputModel> listOutputModel = new List<OutputModel>();
            foreach (var mydict in mydictionary)
            {
                string text = mydict.text.Replace('"', ' ');
                string result = PythonExecute(text);

                List<Intent> IntentsResult = new List<Intent>();
                var intents = result.Split(new string[] { "###" }, StringSplitOptions.None);
                //Окей, тут конверт треба написати в залежності від входу (не єбу який він)
                foreach (var c in intents)
                {
                    if (!string.IsNullOrEmpty(c))
                    {
                        IntentsResult.Add(new Intent { startsAt = text.IndexOf(c), text = c });
                    }
                }

                listOutputModel.Add(new OutputModel { id = mydict.id, intents = IntentsResult });
            }
            return Json(listOutputModel, JsonRequestBehavior.AllowGet);
        }


        [NonAction]
        private string GetGUID(string filename)
        {
            Guid g;
            g = Guid.NewGuid();
            string[] mass = filename.Split('.');
            string NewFileName = g.ToString() + '.' + mass[mass.Length - 1];
            return NewFileName;
        }

    }
}