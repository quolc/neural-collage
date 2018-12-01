// agh.sprintf.js
/* ----------------------------------------------------------------------------

 Author: K. Murase (akinomyoga)

 Changes

 * 2015-05-29 KM created git repository
 * 2014-12-25 KM 様々な言語での実装を確認
 * 2013-09-05 KM added descriptions
 * 2013-09-01 KM first version
 * 2013-09-01 KM created

 ------------------------------------------------------------------------------

 License: The MIT License (MIT)

 Copyright (c) 2013-2015 K. Murase (akinomyoga)

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.

 ----------------------------------------------------------------------------*/

/**
 *  @section sprintf.format 書式指定
 *    書式は以下の形式で指定する。
 *    '%' \<pos\>? \<flag\> \<width\> \<precision\>? \<type\>? \<conv\>
 *
 *    位置指定子   \<pos\>       は引数の番号を指定する。
 *    フラグ       \<flag\>      は出力の見た目を細かく指定する。
 *    幅           \<width\>     は出力時の最小文字数を指定する。
 *    精度         \<precision\> は内容をどれだけ詳しく出力するかを指定する。
 *    サイズ指定子 \<type\>      は引数のサイズ・型を指定する。
 *    変換指定子   \<conv\>      は出力する形式を指定する。
 *
 *  @subsection sprintf.format.pos 位置指定子 (POSIX)
 *    位置指定子は以下の形式を持つ。
 *    \<pos\> := /\d+\$/
 *    整数で引数の番号を指定する。書式指定文字列の次に指定した引数の番号が 1 である。
 *
 *  @subsection sprintf.format.flag フラグ
 *    フラグは以下の形式を持つ。
 *    \<flag\> := ( /[-+ 0#']/ | /\=./ ) +
 *
 *    '-'  (標準)    左寄せを意味する。既定では右寄せである。
 *    '+'  (標準)    非負の数値に正号を付ける事を意味する。
 *    '#'  (標準)    整数の場合、リテラルの基数を示す接頭辞を付ける。
 *                   但し、値が 0 の時は接頭辞を付けない。
 *                   conv = o, x, X のそれぞれに対して "0", "0x", "0X" が接頭辞となる。
 *
 *                   浮動小数点数 conv = f, F, e, E, g, G, a, A の場合は、
 *                   整数 (precision = 0) に対しても小数点を付ける (例 "123.") 事を意味する。
 *                   conv = g, G については小数末尾の 0 の連続を省略せずに全て出力する。
 *    ' '  (標準)    非負の数値の前に空白を付ける事を意味する。
 *                   これは出力幅が width で指定した幅を超えても必ず出力される空白である。
 *    '0'  (標準)    左側の余白を埋めるのに 0 を用いる。但し、空白と異なり 0 は符号や基数接頭辞の右側に追加される。
 *    "'"  (SUSv2)   conv = d, i, f, F, g, G の整数部で桁区切 (3桁毎の ",") を出力する事を意味する。
 *                   但し、flag 0 で指定される zero padding の部分には区切は入れない。
 *
 *    参考: 野良仕様で以下の様なものもあるが、ここでは実装しない。
 *    '=?' (strfmon) 余白に使う文字の指定
 *    ','  (Python)  桁区切。→ "'" (SUSv2) に同じ
 *                   (+ Java, Python-3.1)
 *    '<'  (Python)  左寄せ。'?<' として余白文字を指定できる。
 *                   → '-' (標準) に同じ
 *    '>'  (Python)  右寄せ。'?>' として余白文字を指定できる。
 *                   → 既定 (標準)
 *    '^'  (Python)  中央揃え。'?^' として余白文字を指定できる。
 *    '='  (Python)  整数型において、符号の後に padding を挿入する
 *                   →これは %.3d 等とする事に等価である。
 *    '-'  (Python)  負号のみ表示 → 既定 (標準)
 *    "'?" (PHP)     余白文字の指定。
 *    '('  (Java)    負の数を "()" で括る。
 *
 *  @subsection sprintf.format.width 幅指定子
 *    幅指定子は以下の形式を持つ。
 *    \<width\> := /\d+/ | '*' | '*' /\d+/ '$'
 *
 *    /\d+/         (標準)  最小幅を整数で指定する。
 *    '*'           (標準)  次の引数を読み取って最小幅とする。
 *    '*' /\d+/ '$' (POSIX) 指定した番号の引数を最小幅とする。
 *
 *  @subsection sprintf.format.precision 精度指定子
 *    精度指定子は以下の形式を持つ。
 *    \<precision\> := /\d+/ | '*' | '*' /\d+/ '$'
 *
 *    /\d+/         (標準)  精度を整数で指定する。
 *    '*'           (標準)  次の引数を読み取って精度とする。
 *    '*' /\d+/ '$' (POSIX) 指定した番号の引数を精度とする。
 *
 *    整数の場合は精度で指定した桁だけ必ず整数を出力する。例えば、精度 4 の場合は "0001" など。
 *    精度を指定した時はフラグで指定した '0' は無視される。
 *
 *    浮動小数点数 conv = f, F, e, E, a, A の場合は小数点以下の桁数を指定する。
 *    浮動小数点数 conv = g, G の場合は有効桁数を指定する。
 *    conv = f, F, e, E, g, G に対しては既定値は 6 である。
 *    conv = a, A については倍精度浮動小数点数の16進桁数である 13 が既定値である。
 *
 *    文字列の場合は最大出力文字数を指定する。この文字数に収まらない部分は出力されずに無視される。
 *
 *  @subsection sprintf.format.type サイズ指定子
 *    サイズ指定子は以下の何れかである。
 *
 *          + --------- 整数 ---------- + ------- 浮動小数点数 ------- + -------- 文字 ---------- +
 *          | 本来      (規格)  実装 註 | 本来        (規格)   実装 註 | 本来    (規格)   実装 註 |
 *    ----- + ------------------------- + ---------------------------- + ------------------------ +
 *    既定  | int       (標準) double#1 | 既定        (標準) double    | 既定    (標準) unicode   |
 *    'hh'  | char      (C99)    8bit   | float       (agh)   float#4  | char    (agh)    ascii   |
 *    'h'   | short     (標準)  16bit   | float       (agh)   float#4  | char    (MSVC)   ascii   |
 *    'l'   | long      (標準)  32bit   | double      (C99)  double    | wint_t  (C99)  unicode#6 |
 *    'll'  | long long (C99)   32bit   | long double (agh)  double#5  | --      --          --   |
 *    't'   | ptrdiff_t (C99)   32bit#2 | --          --         --    | --      --          --   |
 *    'z'   | size_t    (C99)   32bit#2 | --          --         --    | --      --          --   |
 *    'I'   | ptrdiff_t (MSVC)  32bit#2 | --          --         --    | --      --          --   |
 *          | size_t                    | --          --         --    | --      --          --   |
 *    'I32' | 32bit     (MSVC)  32bit   | --          --         --    | --      --          --   |
 *    'q'   | 64bit     (BSD)   64bit#3 | --          --         --    | --      --          --   |
 *    'I64' | 64bit     (MSVC)  64bit#3 | --          --         --    | --      --          --   |
 *    'j'   | intmax_t  (C99)  double#1 | --          --         --    | --      --          --   |
 *    'L'   | --        --         --   | long double (標準) double#5  | --      --          --   |
 *    'w'   | --        --         --   | long double (agh)  double#5  | wchar_t (MSVC) unicode   |
 *    ----- + ------------------------- + ---------------------------- + ------------------------ +
 *
 *    #1 JavaScript の数値は内部的には double なので、
 *       サイズ指定子を省略した場合はこの double で表現される整数として変換を行う。
 *    #2 JavaScript で 64bit 整数は厳密に取り扱う事が出来ないので、32bit を native な整数とする。
 *    #3 JavaScript では 64bit 整数は厳密に取り扱う事が出来ない。
 *       取り敢えず 64bit 整数として出力はするものの、巨大な整数では下の方の桁が正確ではない。
 *
 *    #4 規格にないが独自拡張で、h/hh が指定された時は float の精度に落としてから出力する。
 *       (C 言語では float の可変長引数は double に変換されるからそもそも float で指定できない)。
 *    #5 規格上は long double だが JavaScript では long double は取り扱えないので、
 *       double と同じ取扱とする。
 *
 *    #6 POSIX を見ると %lc の引数は wchar_t[2] { wint_t(), null } と書かれている気がする? [要確認]
 *
 *    参考: 以下の様な野良サイズ指定子もある。
 *    'n'  (Ocaml)    native int
 *
 *  @subsection sprintf.format.conv 変換指定子
 *    引数の型及び出力の形式を指定する。以下の何れかの値を取る。
 *
 *    'd', 'i' (標準) 10進符号付き整数
 *    'o'      (標準)  8進符号無し整数
 *    'u'      (標準) 10進符号無し整数
 *    'x', 'X' (標準) 16進符号無し整数     (lower/upper case, 0xa/0XA など)
 *    'f', 'F' (標準) 浮動小数点数         (lower/upper case, inf/INF など)
 *    'e', 'E' (標準) 浮動小数点数指数表記 (lower/upper case, 1e+5/1E+5 など)
 *    'g', 'G' (標準) 浮動小数点数短い表記 (lower/upper case, 1e+5/1E+5 など)
 *    'a', 'A' (C99)  浮動小数点数16進表現 (lower/upper case, 1p+5/1P+5 など)
 *    'c'      (標準) 文字
 *         'C' (XSI)  文字   (本来 wchar_t 用だがこの実装では c と区別しない)
 *    's'      (標準) 文字列
 *         'S' (XSI)  文字列 (本来 wchar_t 用だがこの実装では s と区別しない)
 *    'p'      (標準) ポインタ値。この実装では upper hexadecimal number で出力。
 *    'n'      (標準) 今迄の出力文字数を value[0] に格納
 *    '%'      (標準) "%" を出力
 *
 *    参考: 野良仕様で以下の様なものもあるが、ここでは実装しない。
 *    'b'      (Ruby)   2進符号付き整数。(+ Python, Perl, PHP, D, Haskell)
 *                      (Go) では浮動小数点数に対して decimalless scientific notation with 2進指数
 *    'B'      (Ruby)   2進符号付き整数。(+ Python, Perl)
 *    'n'      (Python) 数値。ロカールに従った出力を行う(桁区切など)。
 *    '%'      (Python) 百分率。数値を百倍して解釈し 'f' 変換指定子で出力する。
 *    'D'      (Perl)   'ld' に同じ。
 *    'U'      (Perl)   'lu' に同じ。
 *    'O'      (Perl)   'lo' に同じ。
 *    'U'      (Go)     'U+%04d' に同じ。Unicode code point の為。
 *    't'      (Go)     true/false
 *    'b', 'B' (Java)   true/false (+ OCaml)
 *    'h', 'H' (Java)   null/'%x'
 *
 *    'q'      (Bash)   文字/文字列をリテラル (quoted string) として出力。 (+ Go Lua)
 *                      Go では文字(整数)は '' で、文字列は "" で囲む。Lua では '' で囲む。
 *    'C'      (OCaml)  文字リテラル '' として出力
 *    'S'      (OCaml)  文字列リテラル "" として出力
 *    'T'      (Go)     typename
 *    'v'      (Go)     default format (+ Haskell)
 *    'p'      (Ruby)   Object#inspect の結果を載せる。
 *
 *    'n'      (Java)   改行
 *    'a', 't' (OCaml)  ? (二つ引数を取って、引数2を引数1に対し出力?)
 *    ','      (OCaml)  何も出力しない
 *    '@'      (OCaml)  '@' を出力。
 *    '!'      (OCaml)  出力先を flush する。
 *
 *  @subsection sprintf.format.ref References
 *    Wikipedia en    <a href="https://en.wikipedia.org/wiki/Printf_format_string">printf format string - Wikipedia</a>
 *    POSIX printf(1) <a href="http://pubs.opengroup.org/onlinepubs/9699919799/utilities/printf.html">printf</a>
 *    POSIX printf(1) <a href="http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap05.html#tag_05">File Format Notation</a>
 *    POSIX printf(3) <a href="http://pubs.opengroup.org/onlinepubs/9699919799/functions/printf.html">fprintf</a>
 *    MSC   位置指定子 <a href="http://msdn.microsoft.com/ja-jp/library/bt7tawza(v=vs.90).aspx">printf_p の位置指定パラメータ</a>
 *    MSC   サイズ指定子 <a href="http://msdn.microsoft.com/ja-jp/library/vstudio/tcxf1dw6.aspx">サイズ指定</a>
 *
 *    Python   [[6.1. string ? 一般的な文字列操作 ? Python 3.4.2 ドキュメント>http://docs.python.jp/3/library/string.html#format-specification-mini-language]]
 *    PHP      [[PHP: sprintf - Manual>http://php.net/manual/ja/function.sprintf.php]]
 *    Perl     [[sprintf - perldoc.perl.org>http://perldoc.perl.org/functions/sprintf.html]]
 *    D言語    [[std.format - D Programming Language - Digital Mars>http://dlang.org/phobos/std_format.html#format-string]]
 *    Lua      [[Lua 5.1 Reference Manual>http://www.lua.org/manual/5.1/manual.html#pdf-string.format]]
 *    Haskell  [[Text.Printf>http://hackage.haskell.org/package/base-4.7.0.2/docs/Text-Printf.html]]
 *    Go       [[fmt - The Go Programming Language>http://golang.org/pkg/fmt/]]
 *    Java     [[Formatter (Java Platform SE 7 )>http://docs.oracle.com/javase/7/docs/api/java/util/Formatter.html#syntax]]
 *    R (wrapper for C printf) [[sprintf {base} | inside-R | A Community Site for R>http://www.inside-r.org/r-doc/base/sprintf]]
 *    OCaml    [[Printf>http://caml.inria.fr/pub/docs/manual-ocaml/libref/Printf.html]]
 */

/*
 * その他の書式指定の文法について
 * - terminfo の setaf 等で使われている文法について
 *   %? %t %e %; 条件分岐
 *   %| %! %- %< 演算子
 *   %{ ... } リテラル
 * - date コマンド
 * - strfmod
 * - zsh PS1
 * - GNU screen
 */

// 実装:
// * 解析は正規表現を使えば良い。
// * 出力結果の構成
//   <左余白> <符号> <ゼロ> <中身> <右余白>
//   計算の順番としては、<符号>+<中身> のペアを決定してから、<ゼロ> または <余白> を付加すれば良い。

(function(__exports) {
  // 本来 this にグローバルオブジェクトが入るはず。
  // もし空だった場合はこのスクリプトを呼び出したときの this を入れる。
  var __global = this || __exports || {};
  if (typeof __global.agh !== 'undefined')
    __exports = __global.agh;
  else if (typeof __global.module !== 'undefined' && typeof __global.module.exports != 'undefined')
    __exports = __global.module.exports;
  else if (__exports == null)
    __exports = __global.agh = {};

  function repeatString(s, len) {
    if (len <= 0) return "";
    var ret = "";
    do if (len & 1) ret += s; while ((len >>= 1) >= 1 && (s += s));
    return ret;
  }

  //---------------------------------------------------------------------------
  // サイズ指定子達

  var INT32_MOD = 0x100000000;
  var INT32_MIN = -0x80000000;
  var INT32_MAX = +0x7FFFFFFF;
  var INT64_MOD = INT32_MOD * INT32_MOD;
  var INT64_MIN = -INT64_MOD / 2.0;
  var INT64_MAX = -INT64_MIN - 1;

  function roundTowardZero(value) {
    return value < 0 ? Math.ceil(value) : Math.floor(value);
  }

  function getIntegerValue(value, type) {
    // 整数は内部的には double で表現されている。
    // ビット演算は 32bit 符号付整数として実行される。
    value = roundTowardZero(value);
    switch (type) {
    case 'hh': // C99 char (8bit signed)
      value &= 0xFF;
      return value >= 0x80 ? value - 0x100 : value;
    case 'h': // short (16bit signed)
      value &= 0xFFFF;
      return value >= 0x8000 ? value - 0x10000 : value;
    case 'l':   // C89  long (32bit signed) (ビット演算を使うと変になる)
    case 'z':   // C99  size_t
    case 't':   // C99  ptrdiff_t
    case 'I32': // MSVC __int32
    case 'I':   // MSVC ptrdiff_t/size_t
      value %= INT32_MOD;
      if (value < INT32_MIN)
        value += INT32_MOD;
      else if (value > INT32_MAX)
        value -= INT32_MOD;
      return value;
    case 'll':  // C99 long long (64bit signed)
    case 'I64': // MSVC __int32
    case 'q':   // BSD  quad word
    case 'L':   // agh exntesion
      value %= INT64_MOD;
      if (value < INT64_MIN)
        value += INT64_MOD;
      else if (value > INT64_MAX)
        value -= INT64_MOD;
      return value;
    case 'j': default: // 変換無し (double の整数)
      return value;
    }
  }

  function getUnsignedValue(value, type) {
    // 整数は内部的には double で表現されている。
    // ビット演算は 32bit 符号付整数として実行される。
    value = roundTowardZero(value);
    switch (type) {
    case 'hh': // 8bit unsigned
      value &= 0xFF;
      return value;
    case 'h': // 16bit unsigned
      value &= 0xFFFF;
      return value;
    case 'l':   // C89  long (32bit unsigned) (ビット演算を使うと変になる)
    case 'z':   // C99  size_t
    case 't':   // C99  ptrdiff_t
    case 'I32': // MSVC __int32
    case 'I':   // MSVC ptrdiff_t/size_t
      value %= INT32_MOD;
      return value < 0 ? value + INT32_MOD : value;
    case 'll':  // C99 long long (64bit unsigned)
    case 'I64': // MSVC __int32
    case 'q':   // BSD  quad word
    case 'L':   // agh exntesion
      value %= INT64_MOD;
      return value < 0 ? value + INT64_MOD : value;
    case 'j': default: // double の整数の 2 の補数??
      if (value < 0) {
        // 例 -0x80 ~ -0x41 → 7 ~ 6+epsilon → nbits = 8, mod = 0x100
        var nbits = value < INT32_MIN ? 1 + Math.ceil(Math.LOG2E * Math.log(-value)) : 32;
        var mod = Math.pow(2, nbits);
        value += mod;
      }
      return value;
    }
  }

  function getFloatValue(value, type) {
    if (type === 'h' || type === 'hh') {
      var sgn = value < 0 ? -1 : 1;
      var exp = Math.floor(Math.LOG2E * Math.log(sgn * value));
      var scale = Math.pow(2, exp - 23); // float (exp = 0) は小数点以下2進23桁まで有効
      value = (0 | value / scale) * scale;
    }

    return value;
  }

  function getCharValue(value, type) {
    value |= 0;
    if (type === 'h' || type === 'hh') {
      value &= 0xFF;
    }
    return value;
  }

  //---------------------------------------------------------------------------
  // 変換指定子達

  /**
   * @section \<conv\>
   *   引数の型及び出力の形式を指定します。
   * - 'd', 'i' 10進符号付き整数
   * - 'o'       8進符号無し整数
   * - 'u'      10進符号無し整数
   * - 'x', 'X' 16進符号無し整数
   */

  var groupIntegerRegs = [
    /(...)(?!$)/g,
    /(^.|...)(?!$)/g,
    /(^..|...)(?!$)/g
  ];
  function groupInteger(text, flag) {
    if (text.length < 4 || !/\'/.test(flag))
      return text;
    else
      return text.replace(groupIntegerRegs[text.length % 3], "$1,");
  }

  var xdigits = "0123456789abcdef";
  function convertInteger(value, flag, precision, base) {
    var out = '';
    do {
      out = xdigits.charAt(value % base) + out;
      value = Math.floor(value / base);
    } while (value > 0);

    if (precision != null)
      out = repeatString('0', precision - out.length) + out;

    return out;
  }
  function convertDecimal(value, flag, precision, type) {
    return groupInteger(convertInteger(value, flag, precision, 10), flag);
  }
  function convertOctal(value, flag, precision, type) {
    return convertInteger(value, flag, precision, 8);
  }
  function convertLowerHex(value, flag, precision, type) {
    return convertInteger(value, flag, precision, 16);
  }
  function convertUpperHex(value, flag, precision, type) {
    return convertInteger(value, flag, precision, 16).toUpperCase();
  }

  /**
   * - 'f', 'F' 浮動小数点数         (lower/upper case, inf/INF など)
   * - 'e', 'E' 浮動小数点数指数表記 (lower/upper case, 1e+5/1E+5 など)
   * - 'g', 'G' 浮動小数点数短い表記 (lower/upper case, 1e+5/1E+5 など)
   * - 'a', 'A' 浮動小数点数16進表現 (lower/upper case, 1p+5/1P+5 など)
   */

  var logTable = [];
  logTable[ 2] = Math.LOG2E;
  logTable[10] = Math.LOG10E;
  logTable[16] = Math.LOG2E / 4;
  function frexp(value, base) {
    if (value === 0) return [0, 0];

    var exp = 1 + Math.floor(logTable[base] * Math.log(value));
    value = value * Math.pow(base, -exp);

    // 際どいずれが起きるので補正
    if (value * base < 1) {
      value *= base;
      exp--;
    } else if (value >= 1) {
      value /= base;
      exp++;
    }

    return [value, exp];
  }

  var regCarryReach = []; // 末尾の 9 の並び, 繰上到達距離測定用
  regCarryReach[10] = /9*$/;
  regCarryReach[16] = /f*$/;
  function generateFloatingSequence(value, precision, base) {
    // value [0, 1) の数値
    var seq = '';
    while (--precision > 0)
      seq += xdigits.charAt(0 | (value = value * base % base));

    // 最後の数字は四捨五入
    // (0-10 の整数になるので繰り上がり処理が必要)
    var last = Math.round(value * base % base);
    if (last == base) {
      var cd = regCarryReach[base].exec(seq)[0].length;
      if (cd < seq.length) {
        // 繰り上がり
        var iinc = seq.length - cd - 1;
        seq = seq.slice(0, iinc) + xdigits.charAt(1 + (0 | seq.charAt(iinc))) + repeatString('0', cd + 1);
      } else {
        // 全て 9 の時 → exp更新, seq = 1000...
        seq = '1' + repeatString('0', cd + 1);
      }
    } else {
      seq += xdigits.charAt(last);
    }
    return seq;
  }

  function omitTrailingZero(text, flag) {
    return text.replace(/(\.[\da-f]*?)0+$/, function($0, $1) {
      if ($1 && $1.length > 1)
        return $1;
      else
        return /#/.test(flag) ? '.' : '';
    });
  }
  function omitTrailingZeroE(text, flag) {
    return text.replace(/(\.\d*?)0+e/, function($0, $1) {
      if ($1 && $1.length > 1)
        return $1 + 'e';
      else
        return /#/.test(flag) ? '.e' : 'e';
    });
  }

  function convertScientific(value, flag, precision, type) { // conv = e E
    if (isNaN(value))
      return 'nan';
    else if (!isFinite(value))
      return 'inf';

    if (precision == null) precision = 6;

    var buff = frexp(value, 10);
    var fr = buff[0], exp = buff[1] - 1;
    var man = generateFloatingSequence(fr, 1 + precision, 10);
    if (man.length > precision + 1) {
      // 99..99 から 100..00 に繰り上がった時
      man = man.slice(0, -1);
      exp++;
    }

    if (precision > 0 || /#/.test(flag))
      man = man.slice(0, 1) + '.' + man.slice(1);

    if (exp < 0)
      exp = 'e-' + (1000 - exp).toString().slice(1);
    else
      exp = 'e+' + (1000 + exp).toString().slice(1);
    return man + exp;
  }
  function convertScientificHex(value, flag, precision, type) { // conv = a A
    if (isNaN(value))
      return 'nan';
    else if (!isFinite(value))
      return 'inf';

    if (precision == null)
      precision = type === 'h' || type === 'hh' ? 6 : 13;

    var buff = frexp(value, 2);
    var fr = buff[0], exp = buff[1] - 1;
    var man = generateFloatingSequence((1 / 8) * fr, precision + 1, 16);
    if (man.length > precision + 1) {
      man = man.slice(0, -1);
      exp++;
    }

    if (man.length > 1 || /#/.test(flag))
      man = man.slice(0, 1) + '.' + man.slice(1);

    man = omitTrailingZero(man, flag);

    if (exp < 0)
      exp = 'p-' + (1000 - exp).toString().slice(1);
    else
      exp = 'p+' + (1000 + exp).toString().slice(1);
    return man + exp;
  }
  function convertFloating(value, flag, precision, type) { // conv = f F
    if (isNaN(value))
      return 'nan';
    else if (!isFinite(value))
      return 'inf';

    if (precision == null) precision = 6;

    if (value >= 1.0) {
      var buff = frexp(value, 10);
      var fr = buff[0], exp = buff[1];
    } else {
      var fr = value / 10, exp = 1;
    }

    var man = generateFloatingSequence(fr, exp + precision, 10);
    if (precision > 0 || /#/.test(flag)) {
      var point = man.length - precision;
      man = groupInteger(man.slice(0, point), flag) + '.' + man.slice(point);
    } else
      man = groupInteger(man, flag);

    return man;
  }
  function convertCompact(value, flag, precision, type) { // conv = g G
    if (isNaN(value))
      return 'nan';
    else if (!isFinite(value))
      return 'inf';

    if (precision == null)
      precision = 6;
    else if (precision < 1)
      precision = 1;

    if (value < 1e-4 || Math.pow(10, precision) <= value + 0.5) {
      // scientific
      var result = convertScientific(value, flag, precision - 1, type);
      if (/#/.test(flag))
        return result;
      else
        return omitTrailingZeroE(result, flag);
    } else {
      // floating point
      var buff = frexp(value, 10);
      var fr = buff[0], exp = buff[1];

      if (precision < 1) precision = 1;
      var man = generateFloatingSequence(fr, precision, 10);
      var point = man.length - (precision - exp); // 小数点挿入位置。末端からの位置が繰り上がり不変。
      // assert(exp <= precision);
      // assert(man.length == precision || man.length == precision + 1)
      // assert(point <= man.length);
      if (point > 0) {
        if (point < man.length || /#/.test(flag))
          man = groupInteger(man.slice(0, point), flag) + '.' + man.slice(point, precision);
      } else {
        man = '0.' + repeatString('0', -point) + man.slice(0, precision);
      }

      if (/#/.test(flag))
        return man;
      else
        return omitTrailingZero(man, flag);
    }
  }

  function convertCompactU      (value, flag, precision, type) { return convertCompact      (value, flag, precision, type).toUpperCase(); }
  function convertFloatingU     (value, flag, precision, type) { return convertFloating     (value, flag, precision, type).toUpperCase(); }
  function convertScientificU   (value, flag, precision, type) { return convertScientific   (value, flag, precision, type).toUpperCase(); }
  function convertScientificHexU(value, flag, precision, type) { return convertScientificHex(value, flag, precision, type).toUpperCase(); }

  /**
   * - 'c', 'C' 文字
   * - 's', 'S' 文字列
   * - 'n' 今迄の出力文字数を value[0] に格納
   * - '%' % を出力
   */

  function convertChar(value, flag, precision, type) {
    return String.fromCharCode(value);
  }
  function convertString(value, flag, precision, type) {
    if (value == null)
      value = value === undefined ? '(undefined)' : '(null)';
    else
      value = value.toString();

    if (precision != null)
      value = value.slice(0, precision);

    return value;
  }
  function convertOutputLength(value, flag, precision, type, outputLength) { // conv = n
    value[0] = getIntegerValue(outputLength, type);
    return '';
  }
  function convertEscaped(value, flag, precision, type) { return '%'; }

  //---------------------------------------------------------------------------

  function prefixOctal(flag)      { return /#/.test(flag) ? '0'  : ''; }
  function prefixLHex(flag)       { return /#/.test(flag) ? '0x' : ''; }
  function prefixUHex(flag)       { return /#/.test(flag) ? '0X' : ''; }
  function prefixFloatLHex(flag)  { return '0x'; }
  function prefixFloatUHex(flag)  { return '0X'; }
  function prefixPointerHex(flag) { return '0x'; }

  var conversions = {
    d: {getv: getIntegerValue , integral: true , signed: true , prefix: null            , conv: convertDecimal       },
    i: {getv: getIntegerValue , integral: true , signed: true , prefix: null            , conv: convertDecimal       },
    u: {getv: getUnsignedValue, integral: true , signed: false, prefix: null            , conv: convertDecimal       },
    o: {getv: getUnsignedValue, integral: true , signed: false, prefix: prefixOctal     , conv: convertOctal         },
    x: {getv: getUnsignedValue, integral: true , signed: false, prefix: prefixLHex      , conv: convertLowerHex      },
    X: {getv: getUnsignedValue, integral: true , signed: false, prefix: prefixUHex      , conv: convertUpperHex      },
    e: {getv: getFloatValue   , integral: false, signed: true , prefix: null            , conv: convertScientific    },
    E: {getv: getFloatValue   , integral: false, signed: true , prefix: null            , conv: convertScientificU   },
    f: {getv: getFloatValue   , integral: false, signed: true , prefix: null            , conv: convertFloating      },
    F: {getv: getFloatValue   , integral: false, signed: true , prefix: null            , conv: convertFloatingU     },
    g: {getv: getFloatValue   , integral: false, signed: true , prefix: null            , conv: convertCompact       },
    G: {getv: getFloatValue   , integral: false, signed: true , prefix: null            , conv: convertCompactU      },
    a: {getv: getFloatValue   , integral: false, signed: true , prefix: prefixFloatLHex , conv: convertScientificHex },
    A: {getv: getFloatValue   , integral: false, signed: true , prefix: prefixFloatUHex , conv: convertScientificHexU},
    c: {getv: getCharValue    , integral: false, signed: false, prefix: null            , conv: convertChar          },
    C: {getv: getCharValue    , integral: false, signed: false, prefix: null            , conv: convertChar          },
    s: {getv: null            , integral: false, signed: false, prefix: null            , conv: convertString        },
    S: {getv: null            , integral: false, signed: false, prefix: null            , conv: convertString        },
    p: {getv: getUnsignedValue, integral: false, signed: false, prefix: prefixPointerHex, conv: convertUpperHex      },
    n: {getv: null            , integral: false, signed: false, prefix: null            , conv: convertOutputLength  },
    '%': {noValue: true       , integral: false, signed: false, prefix: null            , conv: convertEscaped       }
  };

  function printf_impl(fmt) {
    // ※arguments の fmt を除いた部分の番号は 1 から始まり、
    //   位置指定子も 1 から始まるので、位置番号はそのまま arguments の添字に指定して良い。

    var args = arguments;
    var aindex = 1;
    var lastIndex = 0;
    var outputLength = 0;
    var output = fmt.replace(/%(?:(\d+)\$)?([-+ 0#']*)(\d+|\*(?:\d+\$)?)?(\.(?:\d+|\*(?:\d+\$)?)?)?(hh|ll|I(?:32|64)?|[hlLjztqw])?(.|$)/g, function($0, pos, flag, width, precision, type, conv, index) {
      outputLength += index - lastIndex;
      lastIndex = index + $0.length;

      if ((conv = conversions[conv]) == null) {
        var ret = '(sprintf:error:' + $0 + ')';
        outputLength += ret.length;
        return ret;
      }

      pos =
        (pos == null || pos === "") ? null :
        pos === '*' ? 0 | args[aindex++] :
        0 | pos;

      width =
        (width == null || width === "") ? 0 :
        width === '*' ? 0 | args[aindex++]:
        width.charAt(0) === '*' ? args[0 | width.slice(1, -1)] :
        0 | width;

      precision =
        (precision == null || precision === "") ? null:
        precision === '.*' ? 0 | args[aindex++]:
        precision.charAt(1) === '*' ? args[0 | precision.slice(2, -1)] :
        0 | precision.slice(1);

      var value = conv.noValue ? null: pos === null ? args[aindex++] : args[pos];
      if (conv.getv) value = conv.getv(value, type);

      var prefix = '';
      if (conv.signed) {
        if (value < 0) {
          prefix = '-';
          value = -value;
        } else if (/\+/.test(flag))
          prefix = '+';
        else if (/ /.test(flag))
          prefix = ' ';
      }
      if (conv.prefix && value != 0)
        prefix += conv.prefix(flag);

      var body = conv.conv(value, flag, precision, type, outputLength);

      var lpad = '', zero = '', rpad = '';
      width -= prefix.length + body.length;
      if (width >= 1) {
        if (/-/.test(flag)) {
          // POSIX に従うと - の方が優先
          rpad = repeatString(' ', width);
        } else if (/0/.test(flag) && (!conv.integral || precision == null)) {
          zero = repeatString('0', width);
        } else
          lpad = repeatString(' ', width);
      }

      var ret = lpad + prefix + zero + body + rpad;
      outputLength += ret.length;
      return ret;
    });
    outputLength += fmt.length - lastIndex;

    return [outputLength, output];
  }

  __exports.sprintf = function sprintf() {
    var result = printf_impl.apply(this, arguments);
    return result[1];
  };

  __exports.vsprintf = function vsprintf(fmt, args) {
    var result = printf_impl.apply(this, [fmt].concat(args));
    return result[1];
  };

  var stdout = null;
  if (__global.agh && __global.printh)
    stdout = function(text) { printh(agh.Text.Escape(text, 'html')); };
  else if (__global.process && __global.process.stdout)
    stdout = function(text) { process.stdout.write(text); };
  else if (__global.console && __global.console.log)
    stdout = function(text) { console.log(text); };
  else
    stdout = function(text) { document.write(text); };

  __exports.printf = function printf() {
    var result = printf_impl.apply(this, arguments);
    stdout(result[1]);
    return result[0];
  };

  if (__global.agh && __global.agh.scripts)
    __global.agh.scripts.register("agh.sprintf.js");

})(this);

// test
//
// printf("%2$d行目のコマンド %1$sは不正です", "hoge", 20);
// printf("%d(1234) %o(2322) %x(4d2)\n", 1234, 1234, 1234);
// printf("%s(abc) %c(x)\n", "abc", 'x'.charCodeAt(0));
// printf("%*d(   10)", 5, 10);
// printf("%.*s(3)", 3, "abcdef");
// printf("%2d( 3) %02d(03)", 3, 3);
// printf("%1$d:%2$.*3$d:%4$.*3$d(15:035:045)\n", 15, 35, 3, 45);
//
// printf("%%d: [%+10d][%+ 10d][% +10d][%d]", 10, 10, 10, 1e10);
// printf("%%u: [%u][%u][%+u][% u][%-u][%+10u][%-10u]", -1, 10, 10, 10, 10, 10, 10);
// printf("%%x,%%u: %x %o", -1, -1);
// printf("%%a: %a %a %a %a %a %a %a %a %a", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
// printf("%%A: %A %.3A %#.3A", 1e10, 1e10, 1e10);
// printf("%%A: %A %A %A %A", 4096, 2048, 1024, 512);
//
// printf("%%e: %#.3e %#.3e %#.3e", 1e1000, 1e100, 1e10);
// printf("%%e: %#.3e %#.6e %#.10e %#.20e\n", 1.234, 1.234, 1.234, 1.234);
// printf("%%e: %#.1e %#.5e %#.10e %#.100e", 9.999, 9.999, 9.999, 9.999);
// printf("%%f: %f %#g %#f %#.g %f %'f", 1, 1, 1, 1, 1e100, 1e100);
// printf("%%g: %.15g %.15g %#.15g", 1, 1.2, 1.2);
// printf("%%g: %g %.g %#g %#.g", 15, 15, 15, 15);
// printf("%%g: %g %g %g %g %g", 1e-1, 1e-3, 1e-4, 1e-5, 9e-5, 9e-4, 1e10);
// printf("%%#g: %#g %#g %#g %#2g %#6g", 0.1, 1e-5, 1e-4, 1e-4, 1e-4);
// printf("%%#.g: %#.2g %#.2g", 1e-4, 1e-5);
// printf("%%.g: %.g %.g %.g; %.1g %.1g %.1g; %.2g %.3g", 0.1, 0.999, 9.999, 9, 9.9, 9.999, 9.999, 9.999)
//
// printf("%%c: [%c][%c][%c]\n", 1e100, 65, 8);
// printf("%05s %05s\n", 123, "aaa");
// printf("%%p: %p", 512);
// printf("pi: [%1$a][%1$g][%1$'20.9g][%1$020.9g][%1$'020.9g][%2$'020.9g]", Math.PI, Math.PI * 1e3);
