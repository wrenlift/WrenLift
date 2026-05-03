// Sample entry point. Open `fmt.wren` to see syntax highlighting,
// hover, and (once the LSP gains them) goto-def + completion in
// action.

import "@hatch:fmt" for Fmt

class App {
  static run() {
    System.print(Fmt.green("hello"))
    System.print(Fmt.bold(Fmt.red("FAIL")))
    System.print(Fmt.padLeft("3", 4))
    System.print(Fmt.yellow("Hello wren!"))

    var num = 42.8
    System.print("%(Fmt.blue("num: "))" + "%(Fmt.bold(num))!")
  }
}

App.run()
