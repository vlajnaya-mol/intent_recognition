﻿using System.Web;
using System.Web.Mvc;

namespace Hackathon_3_task
{
    public class FilterConfig
    {
        public static void RegisterGlobalFilters(GlobalFilterCollection filters)
        {
            filters.Add(new HandleErrorAttribute());
        }
    }
}
