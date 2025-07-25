import json
import os
import re
import shlex
import unittest
from xml.dom import minidom

from avocado.core import exit_codes
from avocado.core.output import TermSupport
from avocado.utils import genio
from avocado.utils import path as utils_path
from avocado.utils import process, script
from selftests.utils import AVOCADO, TestCaseTmpDir, skipUnlessPathExists

# AVOCADO may contain more than a single command, as it can be
# prefixed by the Python interpreter
AVOCADO_QUOTED = "', '".join(shlex.split(AVOCADO))
PERL_TAP_PARSER_SNIPPET = """#!/bin/env perl
use TAP::Parser;

my $parser = TAP::Parser->new( { exec => ['%s', 'run', 'examples/tests/passtest.py',
                                          'examples/tests/errortest.py',
                                          'examples/tests/warntest.py',
                                          '--tap', '-', '--disable-sysinfo',
                                          '--job-results-dir', '%%s'] } );

while ( my $result = $parser->next ) {
        $result->is_unknown && die "Unknown line \\"" . $result->as_string . "\\" in the TAP output!\n";
}
$parser->parse_errors == 0 || die "Parser errors!\n";
$parser->is_good_plan || die "Plan is not a good plan!\n";
$parser->plan eq '1..3' || die "Plan does not match what was expected!\n";
"""
PERL_TAP_PARSER_SNIPPET = PERL_TAP_PARSER_SNIPPET % AVOCADO_QUOTED

PERL_TAP_PARSER_FAILFAST_SNIPPET = """#!/bin/env perl
use TAP::Parser;

my $parser = TAP::Parser->new( { exec => ['%s', 'run', 'examples/tests/failtest.py',
                                          'examples/tests/errortest.py',
                                          'examples/tests/warntest.py',
                                          '--tap', '-', '--failfast',
                                          '--disable-sysinfo', '--job-results-dir',
                                          '%%s'] } );

while ( my $result = $parser->next ) {
        $result->is_unknown && die "Unknown line \\"" . $result->as_string . "\\" in the TAP output!\n";
}
$parser->parse_errors == 0 || die "Parser errors!\n";
$parser->is_good_plan || die "Plan is not a good plan!\n";
$parser->plan eq '1..3' || die "Plan does not match what was expected!\n";
"""
PERL_TAP_PARSER_FAILFAST_SNIPPET = PERL_TAP_PARSER_FAILFAST_SNIPPET % AVOCADO_QUOTED

OUTPUT_TEST_CONTENT = """#!/bin/env python
import sys
import logging

from avocado import Test
from avocado.utils import process

print("top_print")
sys.stdout.write("top_stdout\\n")
logging.getLogger("avocado.test.stdout").info("top_avocado_stdout")
sys.stderr.write("top_stderr\\n")
logging.getLogger("avocado.test.stderr").info("top_avocado_stderr")
process.run("/bin/echo top_process")

class OutputTest(Test):
    def __init__(self, *args, **kwargs):
        super(OutputTest, self).__init__(*args, **kwargs)
        print("init_print")
        sys.stdout.write("init_stdout\\n")
        logging.getLogger("avocado.test.stdout").info("init_avocado_stdout")
        sys.stderr.write("init_stderr\\n")
        logging.getLogger("avocado.test.stderr").info("init_avocado_stderr")
        process.run("/bin/echo init_process")

    def test(self):
        print("test_print")
        sys.stdout.write("test_stdout\\n")
        logging.getLogger("avocado.test.stdout").info("test_avocado_stdout")
        sys.stderr.write("test_stderr\\n")
        logging.getLogger("avocado.test.stderr").info("test_avocado_stderr")
        process.run("/bin/echo -n test_process > /dev/stdout",
                    shell=True)
        process.run("/bin/echo -n __test_stderr__ > /dev/stderr",
                    shell=True)
        process.run("/bin/echo -n __test_stdout__ > /dev/stdout",
                    shell=True)

    def __del__(self):
        print("del_print")
        sys.stdout.write("del_stdout\\n")
        logging.getLogger("avocado.test.stdout").info("del_avocado_stdout")
        sys.stderr.write("del_stderr\\n")
        logging.getLogger("avocado.test.stderr").info("del_avocado_stderr")
        process.run("/bin/echo -n del_process")
"""

OUTPUT_MODE_NONE_CONTENT = r"""
import sys

from avocado import Test
from avocado.utils import process


class OutputCheckNone(Test):

    def test(self):
        cmd = "%s -c \"import sys; sys.%%s.write('%%s')\"" % sys.executable
        process.run(cmd % ('stdout', '__STDOUT_DONT_RECORD_CONTENT__'))
        process.run(cmd % ('stderr', '__STDERR_DONT_RECORD_CONTENT__'))
"""

OUTPUT_SHOW_TEST = """
#!/usr/bin/env python3

import sys
import tempfile

from avocado import Test
from avocado.core.job import Job
from avocado.core.suite import TestSuite


class PassTest(Test):
    def test1(self):
        config = {'run.results_dir': self.workdir,
                  'core.show': ['none'],
                  'resolver.references': ['/bin/true']}
        suite = TestSuite.from_config(config)
        with Job(config, [suite]) as j:
            j.run()

    def test2(self):
        config = {'run.results_dir': self.workdir,
                  'core.show': ['app'],
                  'resolver.references': ['/bin/true']}
        suite = TestSuite.from_config(config)
        with Job(config, [suite]) as j:
            j.run()

    def test3(self):
        config = {'run.results_dir': self.workdir,
                  'core.show': ['none'],
                  'resolver.references': ['/bin/true']}
        suite = TestSuite.from_config(config)
        with Job(config, [suite]) as j:
            j.run()


if __name__ == '__main__':
    test_results_dir = tempfile.TemporaryDirectory()

    config = {'run.results_dir': test_results_dir.name,
              'resolver.references': [__file__],
              'core.show': ['app']}
    suite = TestSuite.from_config(config)
    with Job(config, [suite]) as j:
        exit_code = j.run()
        test_results_dir.cleanup()
        sys.exit(exit_code)
"""


def perl_tap_parser_uncapable():
    return os.system("perl -e 'use TAP::Parser;'") != 0


def missing_binary(binary):
    try:
        utils_path.find_command(binary)
        return False
    except utils_path.CmdNotFoundError:
        return True


class OutputTest(TestCaseTmpDir):
    def test_log_to_debug(self):
        test = script.Script(
            os.path.join(self.tmpdir.name, "output_test.py"), OUTPUT_TEST_CONTENT
        )
        test.save()
        result = process.run(
            f"{AVOCADO} run "
            f"--job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo --json - -- {test}"
        )
        res = json.loads(result.stdout_text)
        logfile = res["tests"][0]["logfile"]
        # Today, process.run() calls are not part of the test stdout or stderr.
        # Instead those are registered as part of avocado.utils.process logging
        # system. Let's just add a "DEBUG| " to make sure this will not get
        # confused with [stdout].
        expected = [
            b" DEBUG| [stdout] top_process",
            b" DEBUG| [stdout] init_process",
            b" DEBUG| [stdout] test_process",
            b" DEBUG| [stderr] __test_stderr__",
            b" DEBUG| [stdout] __test_stdout__",
        ]

        self._check_output(logfile, expected)

    def _check_output(self, path, exps):
        i = 0
        end = len(exps)
        with open(path, "rb") as output_file:  # pylint: disable=W1514
            output_file_content = output_file.read()
            output_file.seek(0)
            for line in output_file:
                if exps[i] in line:
                    i += 1
                    if i == end:
                        break
            exps_text = "\n".join([exp.decode() for exp in exps])
            error_msg = (
                f"Failed to find message in position {i} "
                f"from\n{exps_text}\n\nin the {path}. Either "
                f"it's missing or in wrong order.\n{output_file_content}"
            )
            self.assertEqual(i, end, error_msg)

    def test_print_to_std(self):
        test = script.Script(
            os.path.join(self.tmpdir.name, "output_test.py"), OUTPUT_TEST_CONTENT
        )
        test.save()
        result = process.run(
            f"{AVOCADO} run "
            f"--job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo --json - -- {test}"
        )
        res = json.loads(result.stdout_text)
        testdir = res["tests"][0]["logdir"]
        exps_full = [
            b"[stdout] top_print",
            b"[stdout] top_stdout",
            b"top_avocado_stdout",
            b"[stderr] top_stderr",
            b"top_avocado_stderr",
            b"[stdout] init_print",
            b"[stdout] init_stdout",
            b"init_avocado_stdout",
            b"[stderr] init_stderr",
            b"init_avocado_stderr",
            b"[stdout] test_print",
            b"[stdout] test_stdout",
            b"test_avocado_stdout",
            b"[stderr] test_stderr",
            b"test_avocado_stderr",
        ]

        exps_stdout = [
            b"top_print",
            b"top_stdout",
            b"top_avocado_stdout",
            b"init_print",
            b"init_stdout",
            b"init_avocado_stdout",
            b"test_print",
            b"test_stdout",
            b"test_avocado_stdout",
        ]
        exps_stderr = [
            b"top_stderr",
            b"top_avocado_stderr",
            b"init_stderr",
            b"init_avocado_stderr",
            b"test_stderr",
            b"test_avocado_stderr",
        ]
        self._check_output(os.path.join(testdir, "debug.log"), exps_full)
        self._check_output(os.path.join(testdir, "stdout"), exps_stdout)
        self._check_output(os.path.join(testdir, "stderr"), exps_stderr)

    @skipUnlessPathExists("/bin/true")
    def test_show(self):
        """
        Checks if `core.show` is respected in different configurations.
        """
        with script.Script(
            os.path.join(self.tmpdir.name, "test_show.py"),
            OUTPUT_SHOW_TEST,
            script.READ_ONLY_MODE,
        ) as test:
            cmd = f"{AVOCADO} run --disable-sysinfo --job-results-dir {self.tmpdir.name} -- {test.path}"
            result = process.run(cmd)
            expected_job_id_number = 1
            expected_bin_true_number = 0
            job_id_number = result.stdout_text.count("JOB ID")
            bin_true_number = result.stdout_text.count("/bin/true")
            self.assertEqual(expected_job_id_number, job_id_number)
            self.assertEqual(expected_bin_true_number, bin_true_number)

    def test_fail_output(self):
        """
        Checks if the test fail messages are not printed to the console.
        """
        cmd = f"{AVOCADO} run --disable-sysinfo --job-results-dir {self.tmpdir.name} -- examples/tests/failtest.py"
        result = process.run(cmd, ignore_status=True)
        self.assertIn(
            "(1/1) examples/tests/failtest.py:FailTest.test: STARTED\n "
            "(1/1) examples/tests/failtest.py:FailTest.test:  FAIL: This "
            "test is supposed to fail",
            result.stdout_text,
        )

    def tearDown(self):
        self.tmpdir.cleanup()


class OutputPluginTest(TestCaseTmpDir):
    def check_output_files(self, debug_log):
        base_dir = os.path.dirname(debug_log)
        json_output_path = os.path.join(base_dir, "results.json")
        self.assertTrue(os.path.isfile(json_output_path))
        with open(json_output_path, "r", encoding="utf-8") as fp:
            json.load(fp)
        xunit_output_path = os.path.join(base_dir, "results.xml")
        self.assertTrue(os.path.isfile(json_output_path))
        try:
            minidom.parse(xunit_output_path)
        except Exception as details:
            xunit_output_content = genio.read_file(xunit_output_path)
            raise AssertionError(
                (
                    f"Unable to parse xunit output: "
                    f"{details}\n\n{xunit_output_content}"
                )
            )
        tap_output = os.path.join(base_dir, "results.tap")
        self.assertTrue(os.path.isfile(tap_output))
        tap = genio.read_file(tap_output)
        self.assertIn("..", tap)
        self.assertIn("\n# debug.log of ", tap)

    def test_output_incompatible_setup(self):
        cmd_line = (
            f"{AVOCADO} run --job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo --xunit - --json - "
            f"examples/tests/passtest.py"
        )
        result = process.run(cmd_line, ignore_status=True)
        expected_rc = exit_codes.AVOCADO_FAIL
        self.assertEqual(
            result.exit_status,
            expected_rc,
            (f"Avocado did not return rc {expected_rc}:" f"\n{result}"),
        )
        error_regex = re.compile(
            r".*Options ((--xunit --json)|"
            "(--json --xunit)) are trying to use stdout "
            "simultaneously.",
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(
            error_regex.match(result.stderr_text),
            (f"Missing error message from output:\n" f"{result.stderr}"),
        )

    def test_output_compatible_setup(self):
        tmpfile = os.path.join(self.tmpdir.name, f"avocado_{__name__}.xml")
        cmd_line = (
            f"{AVOCADO} run --job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo --journal --xunit {tmpfile} "
            f"--json - examples/tests/passtest.py"
        )
        result = process.run(cmd_line, ignore_status=True)
        expected_rc = exit_codes.AVOCADO_ALL_OK
        self.assertEqual(
            result.exit_status,
            expected_rc,
            (f"Avocado did not return rc {expected_rc}:" f"\n{result}"),
        )
        # Check if we are producing valid outputs
        json.loads(result.stdout_text)
        minidom.parse(tmpfile)

    def test_output_compatible_setup_2(self):
        tmpfile = os.path.join(self.tmpdir.name, f"avocado_{__name__}.json")
        cmd_line = (
            f"{AVOCADO} run --job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo --xunit - --json {tmpfile} "
            f"--tap-include-logs examples/tests/passtest.py"
        )
        result = process.run(cmd_line, ignore_status=True)
        expected_rc = exit_codes.AVOCADO_ALL_OK
        self.assertEqual(
            result.exit_status,
            expected_rc,
            (f"Avocado did not return rc {expected_rc}:" f"\n{result}"),
        )
        # Check if we are producing valid outputs
        with open(tmpfile, "r", encoding="utf-8") as fp:
            json_results = json.load(fp)
            debug_log = json_results["debuglog"]
            self.check_output_files(debug_log)
        minidom.parseString(result.stdout_text)

    def test_output_compatible_setup_nooutput(self):
        tmpfile = os.path.join(self.tmpdir.name, f"avocado_{__name__}.xml")
        tmpfile2 = os.path.join(self.tmpdir.name, f"avocado_{__name__}.json")
        # Verify --show=none can be supplied as app argument
        cmd_line = (
            f"{AVOCADO} --show=none run "
            f"--job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo --xunit {tmpfile} --json {tmpfile2} "
            f"--tap-include-logs examples/tests/passtest.py"
        )
        result = process.run(cmd_line, ignore_status=True)
        expected_rc = exit_codes.AVOCADO_ALL_OK
        self.assertEqual(
            result.exit_status,
            expected_rc,
            (f"Avocado did not return rc {expected_rc}:" f"\n{result}"),
        )
        self.assertEqual(result.stdout, b"", f"Output is not empty:\n{result.stdout}")
        # Check if we are producing valid outputs
        with open(tmpfile2, "r", encoding="utf-8") as fp:
            json_results = json.load(fp)
            debug_log = json_results["debuglog"]
            self.check_output_files(debug_log)
        minidom.parse(tmpfile)

    def test_show_test(self):
        cmd_line = (
            f"{AVOCADO} --show=job run "
            f"--job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo examples/tests/passtest.py"
        )
        result = process.run(cmd_line, ignore_status=True)
        expected_rc = exit_codes.AVOCADO_ALL_OK
        self.assertEqual(
            result.exit_status,
            expected_rc,
            (f"Avocado did not return rc {expected_rc}:" f"\n{result}"),
        )
        job_id_list = re.findall("Job ID: (.*)", result.stdout_text, re.MULTILINE)
        self.assertTrue(job_id_list, f"No Job ID in stdout:\n{result.stdout}")
        job_id = job_id_list[0]
        self.assertEqual(len(job_id), 40)

    def test_show_custom_log(self):
        cmd_line = (
            f"{AVOCADO} --show=avocado.test.progress run "
            f"--job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo examples/tests/logging_streams.py"
        )
        result = process.run(cmd_line, ignore_status=True)
        expected_rc = exit_codes.AVOCADO_ALL_OK
        self.assertEqual(
            result.exit_status,
            expected_rc,
            (f"Avocado did not return rc {expected_rc}:" f"\n{result}"),
        )
        expected_tesult = [
            "avocado.test.progress: 1-examples/tests/logging_streams.py:Plant.test_plant_organic:",
            "logging_streams  L0017 INFO | preparing soil on row 0",
            "logging_streams  L0017 INFO | preparing soil on row 1",
            "logging_streams  L0017 INFO | preparing soil on row 2",
            "logging_streams  L0021 INFO | letting soil rest before throwing seeds",
            "logging_streams  L0026 INFO | throwing seeds on row 0",
            "logging_streams  L0026 INFO | throwing seeds on row 1",
            "logging_streams  L0026 INFO | throwing seeds on row 2",
            "logging_streams  L0030 INFO | waiting for Avocados to grow",
            "logging_streams  L0035 INFO | harvesting organic avocados on row 0",
            "logging_streams  L0035 INFO | harvesting organic avocados on row 1",
            "logging_streams  L0035 INFO | harvesting organic avocados on row 2",
            "logging_streams  L0037 ERROR| Avocados are Gone",
        ]
        self.assertTrue(all([x in result.stdout_text for x in expected_tesult]))

    def test_silent_trumps_test(self):
        # Also verify --show=none can be supplied as run option
        cmd_line = (
            f"{AVOCADO} --show=test --show=none run "
            f"--job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo examples/tests/passtest.py"
        )
        result = process.run(cmd_line, ignore_status=True)
        expected_rc = exit_codes.AVOCADO_ALL_OK
        self.assertEqual(
            result.exit_status,
            expected_rc,
            (f"Avocado did not return rc {expected_rc}:" f"\n{result}"),
        )
        self.assertEqual(result.stdout, b"")

    def test_verify_whiteboard_save(self):
        tmpfile = os.path.join(self.tmpdir.name, f"avocado_{__name__}.json")
        config = os.path.join(self.tmpdir.name, "conf.ini")
        content = (
            "[datadir.paths]\nlogs_dir = %s"  # pylint: disable=C0209
            % os.path.relpath(self.tmpdir.name, ".")
        )
        script.Script(config, content).save()
        cmd_line = (
            f"{AVOCADO} --config {config} --show all run "
            f"--job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo examples/tests/whiteboard.py "
            f"--json {tmpfile}"
        )
        result = process.run(cmd_line, ignore_status=True)
        expected_rc = exit_codes.AVOCADO_ALL_OK
        self.assertEqual(
            result.exit_status,
            expected_rc,
            (f"Avocado did not return rc {expected_rc}:" f"\n{result}"),
        )
        with open(tmpfile, "r", encoding="utf-8") as fp:
            json_results = json.load(fp)
            logfile = json_results["tests"][0]["logfile"]
            debug_dir = os.path.dirname(logfile)
            whiteboard_path = os.path.join(debug_dir, "whiteboard")
            self.assertTrue(
                os.path.exists(whiteboard_path),
                f"Missing whiteboard file {whiteboard_path}",
            )
            with open(whiteboard_path, "r", encoding="utf-8") as whiteboard_file:
                self.assertEqual(
                    whiteboard_file.read(),
                    "TXkgbWVzc2FnZSBlbmNvZGVkIGluIGJhc2U2NA==\n",
                    "Whiteboard message is wrong.",
                )

    def test_gendata(self):
        tmpfile = os.path.join(self.tmpdir.name, f"avocado_{__name__}.json")
        cmd_line = (
            f"{AVOCADO} run --job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo "
            f"examples/tests/gendata.py --json {tmpfile}"
        )
        result = process.run(cmd_line, ignore_status=True)
        expected_rc = exit_codes.AVOCADO_ALL_OK
        self.assertEqual(
            result.exit_status,
            expected_rc,
            (f"Avocado did not return rc {expected_rc}:" f"\n{result}"),
        )
        with open(tmpfile, "r", encoding="utf-8") as fp:
            json_results = json.load(fp)
            json_dir = None
            test = json_results["tests"][0]
            if "test_json" in test["id"]:
                json_dir = test["logfile"]

            self.assertTrue(json_dir, "Failed to get test_json output directory")
            json_dir = os.path.join(os.path.dirname(json_dir), "data", "test.json")
            self.assertTrue(
                os.path.exists(json_dir),
                (f"File {json_dir} produced by" f"test does not exist"),
            )

    def test_redirect_output(self):
        redirected_output_path = os.path.join(
            self.tmpdir.name, f"avocado_{__name__}_output"
        )
        cmd_line = (
            f"{AVOCADO} run --job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo examples/tests/passtest.py > {redirected_output_path}"
        )
        result = process.run(cmd_line, ignore_status=True, shell=True)
        expected_rc = exit_codes.AVOCADO_ALL_OK
        self.assertEqual(
            result.exit_status,
            expected_rc,
            f"Avocado did not return rc {expected_rc}:\n{result}",
        )
        self.assertEqual(
            result.stdout,
            b"",
            f"After redirecting to file, output is not empty: {result.stdout}",
        )
        with open(
            redirected_output_path, "r", encoding="utf-8"
        ) as redirected_output_file_obj:
            redirected_output = redirected_output_file_obj.read()
            for code in TermSupport.ESCAPE_CODES:
                self.assertNotIn(
                    code,
                    redirected_output,
                    f"Found terminal support code {code} "
                    f"in redirected output\n{redirected_output}",
                )

    @unittest.skipIf(
        perl_tap_parser_uncapable(), "Uncapable of using Perl TAP::Parser library"
    )
    def test_tap_parser(self):
        with script.TemporaryScript(
            "tap_parser.pl",
            PERL_TAP_PARSER_SNIPPET % self.tmpdir.name,
            self.tmpdir.name,
        ) as perl_script:
            process.run(f"perl {perl_script}")

    @unittest.skipIf(
        perl_tap_parser_uncapable(), "Uncapable of using Perl TAP::Parser library"
    )
    def test_tap_parser_failfast(self):
        with script.TemporaryScript(
            "tap_parser.pl",
            PERL_TAP_PARSER_FAILFAST_SNIPPET % self.tmpdir.name,
            self.tmpdir.name,
        ) as perl_script:
            process.run(f"perl {perl_script}")

    def test_tap_totaltests(self):
        cmd_line = (
            f"{AVOCADO} run examples/tests/passtest.py "
            f"examples/tests/passtest.py examples/tests/passtest.py "
            f"examples/tests/passtest.py --job-results-dir "
            f"{self.tmpdir.name} --disable-sysinfo --tap -"
        )
        result = process.run(cmd_line)
        expr = b"1..4"
        self.assertIn(expr, result.stdout, f"'{expr}' not found in:\n{result.stdout}")

    def test_broken_pipe(self):
        cmd_line = f"({AVOCADO} run --help | whacky-unknown-command)"
        result = process.run(
            cmd_line, shell=True, ignore_status=True, env={"LC_ALL": "C"}
        )
        expected_rc = 127
        self.assertEqual(
            result.exit_status,
            expected_rc,
            (
                f"avocado run to broken pipe did not return "
                f"rc {expected_rc}:\n{result}"
            ),
        )
        self.assertIn(b"whacky-unknown-command", result.stderr)
        self.assertIn(b"not found", result.stderr)
        self.assertNotIn(b"Avocado crashed", result.stderr)

    def test_results_plugins_no_tests(self):
        cmd_line = f"{AVOCADO} run UNEXISTING --job-results-dir {self.tmpdir.name}"
        result = process.run(cmd_line, ignore_status=True)
        self.assertEqual(result.exit_status, exit_codes.AVOCADO_JOB_FAIL)

        xunit_results = os.path.join(self.tmpdir.name, "latest", "results.xml")
        self.assertFalse(os.path.exists(xunit_results))

        json_results = os.path.join(self.tmpdir.name, "latest", "results.json")
        self.assertFalse(os.path.exists(json_results))

        tap_results = os.path.join(self.tmpdir.name, "latest", "results.tap")
        self.assertFalse(os.path.exists(tap_results))

        # Check that no UI output was generated
        self.assertNotIn(b"RESULTS    : PASS ", result.stdout)
        self.assertNotIn(b"JOB TIME   :", result.stdout)

        # Check that plugins do not produce errors
        self.assertNotIn(b"Error running method ", result.stderr)

    def test_custom_log_levels(self):
        cmd_line = (
            f"{AVOCADO} run --store-logging-stream=custom_logger:35 "
            f"--job-results-dir {self.tmpdir.name} "
            f"--disable-sysinfo examples/tests/logging_custom_level.py"
        )
        result = process.run(cmd_line, ignore_status=True)
        expected_rc = exit_codes.AVOCADO_ALL_OK
        self.assertEqual(
            result.exit_status,
            expected_rc,
            (f"Avocado did not return rc {expected_rc}:" f"\n{result}"),
        )
        custom_log_path = os.path.join(
            self.tmpdir.name, "latest", "custom_logger.Level 35.log"
        )
        with open(custom_log_path, "rb") as custom_log_file:
            self.assertIn(b"Custom logger and level message", custom_log_file.read())


if __name__ == "__main__":
    unittest.main()
