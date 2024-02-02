/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT license.
 * See LICENSE in the project root for license information.
 */

/* global document, Office */

Office.onReady((info) => {
  if (info.host === Office.HostType.OneNote) {
    document.getElementById("sideload-msg").style.display = "none";
    document.getElementById("app-body").style.display = "flex";
    document.getElementById("run").onclick = run;
  }
});

async function dumpTextData(context: OneNote.RequestContext, page: OneNote.Page) {
  page.load("title,contents/type,contents/top,contents/left");
  await context.sync();

  var pageContents: OneNote.PageContent[] = page.contents.items;

  const html = "<p><table><tr><th>Paragraph Id</th><th>Type</th><th>Has Sub</th></tr>";
  var itemHtml = "";
  const closingHtml = "</table></p>";

  var itemsAdded = 0;

  for (var pc = 0; pc < pageContents.length; pc++) {
    var pageContent = pageContents[pc];

    if (pageContent.type === "Outline") {
      pageContent.load("outline/paragraphs/type");

      await context.sync();

      var paragraphs: OneNote.Paragraph[] = pageContent.outline.paragraphs.items;

      for (var p = 0; p < paragraphs.length; p++) {
        var paragraph = paragraphs[p];

        paragraph.load("paragraphs");
        await context.sync(); // maybe more efficient to queue these up.

        var subParagraphs = paragraph.paragraphs.items;

        var phtml = "<tr><td>" + paragraph.id + "</td><td>" + paragraph.type + "</td>";

        if (subParagraphs.length > 0) {
          phtml += "<td>yes</td></tr>";
        } else {
          phtml += "<td>no</td></tr>";
        }

        itemHtml += phtml;
        itemsAdded++;
      }
    }
  }

  if (itemsAdded > 0) {
    page.addOutline(80, 320, html + itemHtml + closingHtml);
  } else {
    page.addOutline(80, 320, "<p>No Text Found</p>");
  }

}

async function dumpInkData(context: OneNote.RequestContext, page: OneNote.Page) {
  page.load("title,inkanalysisornull/paragraphs/lines");

  await context.sync();

  var inkAnalysis = page.inkAnalysisOrNull;

  const html = "<p><table><tr><th>Word</th><th>Paragraph Id</th><th>Line Id</th><th>Word Id</th></tr>";
  var itemHtml = "";
  const closingHtml = "</table></p>";

  var wordsAdded = 0;

  if (inkAnalysis && !inkAnalysis.isNullObject) {
    inkAnalysis.load("paragraphs/lines/words");

    await context.sync();

    // note that for..in doesn't seem to work so we use a standard for loop with an index
    var paragraphs: OneNote.Paragraph[] = inkAnalysis.paragraphs.items;
    for (var i = 0; i < paragraphs.length; i++) {
      var paragraph: OneNote.Paragraph = paragraphs[i];

      for (var l = 0; l < paragraph.lines.items.length; l++) {
        var line: OneNote.InkAnalysisLine = paragraph.lines.items[l];

        for (var w = 0; w < line.words.items.length; w++) {
          var word: OneNote.InkAnalysisWord = line.words.items[w];
          word.load("wordAlternates/languageId/strokePointers/inkStrokeId");

          var strokes = "";
          var strokePointers: OneNote.InkStrokePointer[] = word.strokePointers;
          for (var sp = 0; sp < strokePointers.length; sp++) {
            var strokePointer = strokePointers[sp];
            if (strokePointer.inkStrokeId !== undefined) {
              strokes += strokePointer.inkStrokeId + ", ";
            }
          }

          const td1 = "<td>" + word.wordAlternates[0] + "</td>";
          const td2 = "<td>" + paragraph.id + "</td>";
          const td3 = "<td>" + line.id + "</td>";
          const td4 = "<td>" + word.id + "</td>";
          const td5 = "<td>" + strokes + "</td>";
          const waHtml = "<tr>" + td1 + td2 + td3 + td4 + td5 + "</tr>";
          itemHtml += waHtml;

          wordsAdded++;
        }
      }
    }
  }

  if (wordsAdded > 0) {
    page.addOutline(20, 20, html + itemHtml + closingHtml);
  } else {
    page.addOutline(20, 20, "<p>No Handwriting Detected");
  }
}

export async function run() {
  try {
    OneNote.run(async (context) => {
      // Get the current page.
      const page: OneNote.Page = context.application.getActivePageOrNull();

      await dumpInkData(context, page);
      await dumpTextData(context, page);
    });
  } catch (error) {
    //page.addOutline(40, 90, '<p>Error Running Ink Dump Tool: ' + error + '</p>');
  }
}
